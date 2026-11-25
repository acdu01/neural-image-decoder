"""Optional real-browser checks for the generated offline demo (Playwright)."""
from pathlib import Path
import argparse
import json
from playwright.sync_api import sync_playwright


def check(file, output, executable=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    errors, network = [], []
    with sync_playwright() as p:
        options = {"headless": True}
        if executable:
            options["executable_path"] = executable
        browser = p.chromium.launch(**options)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: network.append(request.url) if request.url.startswith(("http:", "https:")) else None)
            page.goto(Path(file).resolve().as_uri())
            page.locator("#experiment").wait_for(state="visible")
            page.wait_for_function("Array.from(document.querySelectorAll('.gallery img')).every(img => img.complete && img.naturalWidth > 0)")
            initial_budget = page.locator("#budgetValue").inner_text()
            assert int(initial_budget) > 1
            assert page.locator(".image-card").count() == 4
            page.locator("#budget").focus()
            page.keyboard.press("Home")
            assert page.locator("#budgetValue").inner_text() == "1"
            assert len(page.locator(".confusion .matrix-cell").all()) == 16
            page.locator("#next").click()
            assert page.locator("#trial").input_value() == "1"
            page.locator("#previous").click()
            assert page.locator("#trial").input_value() == "0"
            last = str(page.locator("#trial option").count() - 1)
            page.locator("#trial").select_option(last)
            assert page.locator("#next").is_disabled()
            page.locator("#trial").select_option("0")
            page.locator("#budget").focus()
            page.keyboard.press("End")
            assert page.locator("#budgetValue").inner_text() == initial_budget
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(output / "desktop.png"))
            page.locator(".methods summary").click()
            assert page.locator("#fullDownload").get_attribute("href").startswith("blob:")
            with page.expect_download() as event:
                page.locator("#fullDownload").click()
            assert event.value.suggested_filename == "results.json"
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("window.scrollTo(0,0)")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(output / "mobile.png"))
            assert not errors, errors
            assert not network, network
            print(json.dumps({"offline_load": True, "images": 4, "slider": True,
                              "trial_navigation": True, "download": True,
                              "mobile_no_overflow": True, "js_errors": errors,
                              "network_requests": network, "screenshots": str(output)}, indent=2))
        finally:
            browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--executable")
    args = parser.parse_args()
    check(args.file, args.output, args.executable)
