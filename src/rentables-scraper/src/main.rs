use dotenv::dotenv;
use scrapegoat::ScrapeGoat;
use scraper::{Html, Selector};
use std::sync::Arc;

use crate::data::db::Db;
use crate::data::{ListingData, parse_field};
use crate::mgt::Mgt;

mod data;
mod mgt;

fn parse_listings(html: &str) -> Vec<ListingData> {
  let html = Html::parse_document(html);

  let selector = Selector::parse(r#"#listings > div > div"#).unwrap();
  let details = Selector::parse(r#"div:nth-child(2) > a > ul > li"#).unwrap();
  let price = Selector::parse("div:nth-child(2) > div > strong").unwrap();
  let link = Selector::parse("div:nth-child(2) > div > a").unwrap();

  html
    .select(&selector)
    .filter_map(|elem| {
      let mut data = elem.select(&details);

      let rooms = data
        .next()?
        .inner_html()
        .trim()
        .split(" ")
        .nth(0)?
        .parse()
        .unwrap_or(0);
      data.next();
      let m_sqrd = data
        .next()?
        .inner_html()
        .trim()
        .split(" ")
        .nth(0)?
        .parse()
        .unwrap_or(0);

      let price_cold = elem
        .select(&price)
        .next()?
        .inner_html()
        .trim()
        .replace(".", "")
        .split(" ")
        .nth(1)?
        .parse()
        .unwrap_or(0);

      let slug = elem
        .select(&link)
        .next()?
        .value()
        .attr("href")?
        .replace("/inserate/", "")
        .into();

      Some(ListingData {
        slug,
        rooms,
        m_sqrd,
        price_cold,
      })
    })
    .collect()
}

#[tokio::main]
async fn main() {
  dotenv().ok();

  let scraper = ScrapeGoat::new("./proxies.txt", "./user_agents.txt", 10).unwrap();
  let db = Arc::new(
    Db::new(&std::env::var("DATABASE_URL").unwrap())
      .await
      .unwrap(),
  );
  let mut mgt = Mgt::new(scraper, 10);

  mgt
    .run(|page, html| {
      let db = db.clone();
      async move {
        let listings = parse_listings(&html);
        for listing in &listings {
          db.insert_listing(listing).await.ok();
        }
        println!("page {}: {} listings", page, listings.len());
      }
    })
    .await;
}
