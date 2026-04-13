use scrapegoat::ScrapeGoat;
use std::{future::Future, sync::Arc};
use tokio::sync::{Mutex, mpsc};

enum JobResult {
  Page { page: u32, html: String },
  NotFound { page: u32 },
}

pub struct Mgt {
  tx: mpsc::Sender<JobResult>,
  rx: mpsc::Receiver<JobResult>,
  in_flight: u32,
  next_page: u32,
  max_concurrent: u32,
  scraper: Arc<Mutex<ScrapeGoat>>,
  done: bool,
}

impl Mgt {
  pub fn new(scraper: ScrapeGoat, max_concurrent: u32) -> Self {
    let (tx, rx) = mpsc::channel(max_concurrent as usize * 2);
    Self {
      tx,
      rx,
      in_flight: 0,
      next_page: 1,
      max_concurrent,
      scraper: Arc::new(Mutex::new(scraper)),
      done: false,
    }
  }

  fn spawn_job(&mut self, page: u32) {
    let tx = self.tx.clone();
    let scraper = self.scraper.clone();

    tokio::spawn(async move {
      let url = format!("https://rentumo.ch/mietobjekte?types=apartment&page={}", page);
      let result = scraper.lock().await.get_page(&url).await;

      let msg = match result {
        Ok(html) => JobResult::Page { page, html },
        Err(e) if e.status == 404 => JobResult::NotFound { page },
        Err(_) => JobResult::NotFound { page },
      };

      tx.send(msg).await.ok();
    });

    self.in_flight += 1;
    self.next_page += 1;
  }

  pub async fn run<F, Fut>(&mut self, mut on_page: F)
  where
    F: FnMut(u32, String) -> Fut,
    Fut: Future<Output = ()>,
  {
    while self.in_flight < self.max_concurrent {
      self.spawn_job(self.next_page);
    }

    while self.in_flight > 0 {
      match self.rx.recv().await {
        Some(JobResult::Page { page, html }) => {
          self.in_flight -= 1;
          on_page(page, html).await;
          if !self.done {
            self.spawn_job(self.next_page);
          }
        }
        Some(JobResult::NotFound { .. }) => {
          self.in_flight -= 1;
          self.done = true;
        }
        None => break,
      }
    }
  }
}
