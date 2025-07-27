from selenium import webdriver

driver = webdriver.Chrome()  # no headless mode
driver.get("https://www.google.com")

print(driver.title)
driver.quit()
