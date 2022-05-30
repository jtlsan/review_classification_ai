import crawling_driver as crawling
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
import sys
import time

page_list_selector = '#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a'
page_btn_root_xpath = '//*[@id="reviewListFragment"]/div[DIV]/div[2]/div/a[NUM]'
review_selector = '#reviewListFragment > div > div.review-contents > div.review-contents__text'
'#reviewListFragment > div:nth-child(6) > div.review-contents > div.review-contents__text'
'#reviewListFragment > div:nth-child(7) > div.review-contents > div.review-contents__text'
page_next_btn_xpath = '//*[@id="reviewListFragment"]/div[11]/div[2]/div/a[8]'
page_next_btn_selector ='#reviewListFragment > div.nslist_bottom > div.pagination.textRight > div > a.fa.fa-angle-right.paging-btn.btn.next'
style_reivew_xpath = '//*[@id="estimate_style"]'
item_review_xpath = '//*[@id="estimate_photo"]'
normal_review_xpath = '//*[@id="estimate_goods"]'
review_types = [style_reivew_xpath, item_review_xpath, normal_review_xpath]
page_cnt_selector = '#reviewListFragment > div.nslist_bottom > div.box_page_msg'

def crawl_by_url(url):
	web = crawling.WebCrawling(
			review_selector,
			page_list_selector,
			page_btn_root_xpath,
			page_next_btn_selector,
			page_cnt_selector,
			30,
			)
	web.set_url(url)
	web.webdriver.set_window_size(1920, 1380)
	for review_type in review_types:
		web.click_move(review_type)
		web.extract_through_pages()

	return web.reviews

if __name__ == '__main__':
	url = 'https://store.musinsa.com/app/goods/858911'
	crawl_by_url(url)
