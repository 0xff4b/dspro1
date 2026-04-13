pub mod db;

use scraper::{Html, Selector};
use std::str::FromStr;

#[derive(Debug, Default)]
pub struct ListingData {
  pub slug: String,
  pub rooms: u16,
  pub m_sqrd: u32,
  pub price_cold: u32,
  // from details page
  // pub address: String,
  // pub desc: String,
  // pub floor: u8,
  // pub auxiliary_costs: u32,
  // pub kaution: u32,
}

pub fn parse_field<T: FromStr + Default, F: Fn(&str) -> String>(
  html: &Html,
  selector: &Selector,
  transform: F,
) -> T {
  html
    .select(selector)
    .next()
    .map(|el| {
      transform(el.inner_html().trim())
        .parse()
        .unwrap_or_default()
    })
    .unwrap_or_default()
}
