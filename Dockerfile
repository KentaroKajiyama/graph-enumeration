FROM rust:1.80-bookworm

# musl で静的リンク
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools clang llvm-dev libclang-dev pkg-config build-essential && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl

WORKDIR /work
# 依存キャッシュ
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main(){}" > src/main.rs && cargo build --release --target x86_64-unknown-linux-musl && rm -rf src
# 本体
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl
