use scraper::ElementRef;

pub struct ListingData {
  pub slug: String,
  pub rooms: Option<u16>,
  pub listing_type: Option<String>,
  pub m_squared: Option<u32>,
  pub price_cold: Option<u32>,
}

impl ListingData {
  pub fn new(elem: ElementRef<'_>) -> ListingData {
    todo!()
  }
}