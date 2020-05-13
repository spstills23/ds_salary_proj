# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:05:07 2020

@author: seans
"""
import glassdoor_scraper as gs
import pandas as pd

path = 'C:/Users/seans/OneDrive/Desktop/Python/ds_salary_proj/chromedriver'
df = gs.get_jobs('data scientist',15,False,path, 15)
df
