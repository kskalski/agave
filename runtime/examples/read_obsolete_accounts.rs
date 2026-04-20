use {
    solana_runtime::snapshot_utils::{
        MAX_OBSOLETE_ACCOUNTS_FILE_SIZE, read_obsolete_accounts_bufreader,
        read_obsolete_accounts_large_file_buf_reader,
    },
    std::{path::Path, time::Instant},
};

const OBSOLETE_ACCOUNTS_PATH: &str = "/mnt/plain/agave_target/tmp/obsolete_accounts";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_bufreader = args.iter().any(|a| a == "--bufreader");

    let path = Path::new(OBSOLETE_ACCOUNTS_PATH);
    let t = Instant::now();

    let result = if use_bufreader {
        println!("reader: BufReader");
        read_obsolete_accounts_bufreader(path, MAX_OBSOLETE_ACCOUNTS_FILE_SIZE)
    } else {
        println!("reader: large_file_buf_reader (io-uring)");
        read_obsolete_accounts_large_file_buf_reader(path, MAX_OBSOLETE_ACCOUNTS_FILE_SIZE)
    };

    match result {
        Ok(count) => println!("entries: {count}, elapsed: {:?}", t.elapsed()),
        Err(e) => eprintln!("error: {e}"),
    }
}
