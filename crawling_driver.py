from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import os
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from pyvirtualdisplay import Display

class WebCrawling:
	def __init__(self, review_selector, page_list_selector, page_btn_root_xpath, page_next_btn_selector, page_cnt_selector, maximum=-1):
		self.webdriver = self.get_webdriver()
		self.review_selector = review_selector
		self.page_list_selector = page_list_selector
		self.page_btn_root_xpath = page_btn_root_xpath
		self.page_next_btn_selector = page_next_btn_selector
		self.page_cnt_selector = page_cnt_selector
		self.reviews = []
		self.errors = []
		self.duplicated = 0
		if maximum < 0:
			self.maximum = 1e10
		else:
			self.maximum = maximum

	def get_webdriver(self):
		chrome_options = webdriver.ChromeOptions()
		chrome_options.add_argument('--headless')
		chrome_options.add_argument('--no-sandbox')
		chrome_options.add_argument("--single-process")
		chrome_options.add_argument("--disable-dev-shm-usage")
		#chrome_options.add_argument('window-size=1920, 1080')
		chrome_options.add_argument('--log-level=3')
		#chrome_options.add_argument('start-maximized')

		# service in Window
		#service = Service('./chromedriver.exe')
		# service in Linux
		#service = Service('./chromedriver')



		# driver in Window
		driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=chrome_options)
		# driver in Linux
		#display = Display(visible=0, size=(1920, 1380))
		#display.start()

		#path = '/home/jtlsan/projects/capstone2/codes/test/review_classification/chromedriver'
		#driver = webdriver.Chrome(path, options=chrome_options)

		return driver


	def extract_review(self, review_list):
		texts = []
		for review in review_list:
			text = review.get_text().strip()
			if not text:
				continue
			text = text.replace('\n', '')
			texts.append(text)
		self.reviews.extend(texts)


	def extract_through_pages(self):
		page_elements = self.webdriver.find_elements(By.CSS_SELECTOR, self.page_list_selector)
		pages = len(page_elements)

		if pages==0:
			return
		
		for i in range(3, pages-1):
			if self.is_review_full():
				return
			page_xpath = self.page_btn_root_xpath.replace('NUM', str(i))
			review_div_cnt = len(self.webdriver.find_elements(By.CSS_SELECTOR, self.review_selector)) + 1
			page_xpath = page_xpath.replace('DIV', str(review_div_cnt))
			page = self.webdriver.find_element(By.XPATH, page_xpath)
			page.click()
			time.sleep(1)
			soup = BeautifulSoup(self.webdriver.page_source, 'html.parser')
			review_list = soup.select(self.review_selector)
			
			self.extract_review(review_list)
		
		page_cnt_elem = soup.select(self.page_cnt_selector)[0]
		page_cnt_msg = page_cnt_elem.get_text().strip()
		page_cnt_tokens = page_cnt_msg.split()
		total = int(page_cnt_tokens[0])
		cur = int(page_cnt_tokens[3])
		if total == cur:
			done = True
		else:
			done = False
			
		if not done:
			#next_btn = self.webdriver.find_element(By.XPATH, page_next_btn_xpath)
			next_btn = self.webdriver.find_element(By.CSS_SELECTOR, self.page_next_btn_selector)
			next_btn.click()
			time.sleep(1)
			self.extract_through_pages()

	def set_url(self, url):
		self.webdriver.get(url)

	def is_review_full(self):
		if self.maximum == len(self.reviews):
			return True
		return False

	def write_texts(self, texts, pbar):
		with open(self.text_path, 'a') as f:
			for text_group in texts:
				combined = ''
				for text in text_group:
					assert text, '빈 텍스트'
					combined += text + ' '
				combined = combined.strip()
				try:
					f.write(combined + '\n')
				except UnicodeEncodeError as e:
					self.errors.append(e)
				#pbar.update(1)

	def click_move(self, xpath):
		self.webdriver.find_element(By.XPATH, xpath).click()
		time.sleep(2)

	def move_href(self, xpath):
		href = self.webdriver.find_element(By.XPATH, xpath).get_attribute('href')
		self.webdriver.get(href)
		time.sleep(2)

	def print_errors(self):
		print(f'found {len(self.errors)} errors')
		print(f'found {self.duplicated} duplications')
		print()
		
		

					
