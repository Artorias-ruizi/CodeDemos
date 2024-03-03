from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from lxml import etree
import time

start = 0
# url = f"https://scholar.google.com/scholar?start={start}&hl=zh-CN&as_sdt=2005&cites=8592846429445647326&scipsc="
url1= f"https://scholar.google.com/scholar?oi=bibs&hl=zh-CN&cites=2297623996522050844&as_sdt=5"
webdriver = Chrome()

# page = int(input("页数: "))
page = 25 # 爬的页数
webdriver.get(url1)

webdriver.maximize_window()
time.sleep(40) # 30s之内验证完 修改完设置
with open('cite.txt', 'a+')as f:
    for _ in range(page):
        for i in range(1, 21):
            # xpath = f'(//div[@class="gs_fl gs_flb"]/a[6])[{i}]'
            xpath = f'(//div[@class="gs_fl gs_flb"]/a[contains(text(),"Bib")])[{i}]'
            url_bib = webdriver.find_element(By.XPATH, xpath)
            try:
                url_bib.click()
            except Exception:
                # print(Exception)
                continue

            html_data = webdriver.page_source
            html_obj = etree.HTML(html_data)
            res = html_obj.xpath('/html/body/pre/text()')
            print(res)
            if res==[]:
                time.sleep(30)
                html_data = webdriver.page_source
                res = html_obj.xpath('/html/body/pre/text()')
            for ress in res:
                f.write(ress + '\n')
            webdriver.back()

        js = "document.documentElement.scrollTop=2000"
        webdriver.execute_script(js)

        next_node = webdriver.find_element(By.XPATH, '//tbody/tr/td[@align="left"]')
        next_node.click()
