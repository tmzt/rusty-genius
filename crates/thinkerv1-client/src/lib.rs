// Copyright 2024-2026 TME
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use async_channel::{self, Receiver, Sender};
use futures::{
    lock::Mutex,
    select,
    FutureExt, Stream,
};
use futures_lite::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use rusty_genius_thinkerv1::{Request, Response};
use smol::{
    net::{TcpStream, unix::UnixStream},
    process::Command,
    spawn,
};
use std::{collections::HashMap, io, net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(thiserror::Error, Debug)]
pub enum ClientError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Client is disconnected")]
    Disconnected,
}
pub type Result<T> = std::result::Result<T, ClientError>;

type ResponseMap = Arc<Mutex<HashMap<String, Sender<Result<Response>>>>>;

pub enum Address {
    Tcp(SocketAddr),
    Uds(PathBuf),
    Command(Command),
}

async fn handle_response(line: &str, responses: &ResponseMap) {
    let response: Result<Response> = serde_json::from_str(line).map_err(Into::into);
    let id = match &response {
        Ok(Response::Status(r)) => Some(r.id.clone()),
        Ok(Response::Event(r)) => Some(match r {
            rusty_genius_thinkerv1::EventResponse::Thought { id, .. } => id.clone(),
            rusty_genius_thinkerv1::EventResponse::Content { id, .. } => id.clone(),
            rusty_genius_thinkerv1::EventResponse::Complete { id, .. } => id.clone(),
            rusty_genius_thinkerv1::EventResponse::Embedding { id, .. } => id.clone(),
        }),
        Err(_) => None,
    };

    if let Some(id_str) = id {
        let mut res_map = responses.lock().await;
        if let Some(sender) = res_map.get(&id_str) {
            let is_final = matches!(&response, Ok(Response::Event(rusty_genius_thinkerv1::EventResponse::Complete {..})) | Ok(Response::Event(rusty_genius_thinkerv1::EventResponse::Embedding {..}))) || matches!(&response, Ok(Response::Status(s)) if s.status == "ready" || s.status == "error");
            if sender.send(response).await.is_err() || is_final {
                res_map.remove(&id_str);
            }
        }
    }
}

async fn connection_loop<S>(stream: S, responses: ResponseMap, writer_rx: Receiver<Vec<u8>>)
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let (reader, mut writer) = futures_lite::io::split(stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        select! {
            read_result = reader.read_line(&mut line).fuse() => match read_result {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if !line.trim().is_empty() {
                        handle_response(&line, &responses).await;
                    }
                    line.clear();
                }
            },
            write_request = writer_rx.recv().fuse() => match write_request {
                Ok(mut data) => {
                    data.push(b'\n');
                    if writer.write_all(&data).await.is_err() { break; }
                }
                Err(_) => break,
            },
        }
    }
    let mut res_map = responses.lock().await;
    for (_, sender) in res_map.drain() {
        let _ = sender.try_send(Err(ClientError::Disconnected));
    }
}

#[derive(Clone)]
pub struct Client {
    writer: Sender<Vec<u8>>,
    responses: ResponseMap,
}

impl Client {
    pub async fn connect(address: Address) -> Result<Self> {
        let responses: ResponseMap = Arc::new(Mutex::new(HashMap::new()));
        let (writer_tx, writer_rx) = async_channel::unbounded();

        match address {
            Address::Tcp(addr) => {
                let stream = TcpStream::connect(addr).await?;
                spawn(connection_loop(stream, responses.clone(), writer_rx)).detach();
            }
            Address::Uds(path) => {
                let stream = UnixStream::connect(path).await?;
                spawn(connection_loop(stream, responses.clone(), writer_rx)).detach();
            }
            Address::Command(mut cmd) => {
                let mut child = cmd.stdin(std::process::Stdio::piped())
                                   .stdout(std::process::Stdio::piped())
                                   .spawn()?;
                let stdin = child.stdin.take().unwrap();
                let stdout = child.stdout.take().unwrap();
                
                let responses_clone = responses.clone();
                spawn(async move {
                    let mut reader = BufReader::new(stdout);
                    let mut line = String::new();
                    loop {
                        line.clear();
                        if reader.read_line(&mut line).await.is_err() { break; }
                        if line.is_empty() { break; }
                        if !line.trim().is_empty() {
                            handle_response(&line, &responses_clone).await;
                        }
                    }
                }).detach();
                
                spawn(async move {
                    let mut writer = stdin;
                    while let Ok(mut data) = writer_rx.recv().await {
                        data.push(b'\n');
                        if writer.write_all(&data).await.is_err() { break; }
                    }
                }).detach();
            }
        };

        Ok(Self {
            writer: writer_tx,
            responses,
        })
    }

    pub async fn request(&self, request: Request) -> Result<impl Stream<Item = Result<Response>>> {
        let id = request.get_id().to_string();
        let (tx, rx) = async_channel::unbounded();
        self.responses.lock().await.insert(id.clone(), tx);

        let data = serde_json::to_vec(&request)?;
        if self.writer.send(data).await.is_err() {
            self.responses.lock().await.remove(&id);
            return Err(ClientError::Disconnected);
        }
        Ok(rx)
    }
}
