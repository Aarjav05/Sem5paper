import pandas as pd
import time
from playwright.sync_api import sync_playwright

def scrape_featured_insights():
    items = []

    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=False)
        page = browser.new_page()

        # Visit the insights page
        try:
            page.goto("https://www.mckinsey.com/featured-insights", timeout=60000)
        except Exception as err:
            print("Error loading page:", err)
            browser.close()
            return items

        # Accept cookie pop-up if it appears
        try:
            page.wait_for_selector("button#onetrust-accept-btn-handler", timeout=5000)
            page.click("button#onetrust-accept-btn-handler")
            print("Cookies accepted")
            time.sleep(1)
        except:
            print("No cookie prompt seen")

        # Scroll a few times to allow dynamic content loading
        for step in range(8):
            page.mouse.wheel(0, 4000)
            time.sleep(2)

        # Collect visible article links
        link_nodes = page.query_selector_all("a.mdc-c-link___lBbY1_4145629")
        print("Found", len(link_nodes), "links")

        for link in link_nodes:
            try:
                headline = link.inner_text().strip()
                href = link.get_attribute("href")

                # Walk upward looking for the eyebrow label
                label = link.evaluate("""
                el => {
                    let anc = el;
                    for (let depth = 0; depth < 5; depth++) {
                        if (!anc) break;
                        const tagBox = anc.querySelector(".mck-c-eyebrow");
                        if (tagBox && tagBox.textContent)
                            return tagBox.textContent.trim();
                        anc = anc.parentElement;
                    }
                    return null;
                }
                """) or "Unknown"

                if href:
                    full_href = href if href.startswith("http") else f"https://www.mckinsey.com{href}"
                    items.append({
                        "Title": headline,
                        "URL": full_href,
                        "Topic": label
                    })
            except Exception as e:
                print("Skipping a link:", e)
                continue

        browser.close()

    return items

def main():
    rows = scrape_featured_insights()
    df = pd.DataFrame(rows)
    df.to_excel("mckinsey_insights.xlsx", index=False)
    print(f"Saved {len(rows)} records to Excel")

if __name__ == "__main__":
    main()
