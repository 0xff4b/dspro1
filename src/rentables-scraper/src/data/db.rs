use crate::data::ListingData;
use sqlx::PgPool;

pub struct Db {
  pool: PgPool,
}

impl Db {
  pub async fn new(url: &str) -> Result<Self, sqlx::Error> {
    let pool = PgPool::connect(url).await?;
    Ok(Self { pool })
  }

  pub async fn insert_listing(&self, listing: &ListingData) -> Result<(), sqlx::Error> {
    sqlx::query!(
      r#"
            INSERT INTO listings (slug, rooms, m_sqrd, price_cold)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (slug) DO NOTHING
            "#,
      listing.slug,
      listing.rooms as i32,
      listing.m_sqrd as i32,
      listing.price_cold as i32,
    )
    .execute(&self.pool)
    .await?;

    Ok(())
  }
}
