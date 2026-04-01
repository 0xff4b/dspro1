use std::{
    fs::{File, read_to_string},
    io::Write,
};

use scrapegoat::ScrapeGoat;
use scraper::{ElementRef, Html, Selector};

mod data;

const PAGE_LIMIT: u16 = 375;
/*
get all mehr sehen links
r#"#listings > div > div > div.bg-white.xl\:px-5.px-4.xl\:pt-3\.5.xl\:pb-3\.5.pt-2\.5.pb-2\.5 > div > a.text-sm.font-bold.block.text-center.text-blue_reg"#

get all listing items
r#"#listings > div"# ?

*/


// #[tokio::main]
// async fn main() {
//   let mut scraper = ScrapeGoat::new("./proxies.txt", "./user_agents.txt", 1).expect("fuck you");

//   let res = scraper
//     .get_page("https://rentumo.ch/mietobjekte?page=0")
//     .await
//     .expect("adwoij");

//   let mut file = File::create("output.txt").expect("Failed to create file");
//   file.write_all(res.as_bytes())
//     .expect("Failed to write to file");

//   println!("{res}");
// }

fn main() {
  let test = read_to_string("./output.html").unwrap();

  let html = Html::parse_document(&test);

  // get all listing links
  let selector = Selector::parse(r#"#listings > div > div"#).unwrap();

  let price_selector = Selector::parse(r#"div:nth-child(2) > a > ul > li"#).unwrap();


  for elem in html.select(&selector) {
    println!("{:?}", elem.select(&price_selector).nth(0).unwrap().inner_html());
  }
}