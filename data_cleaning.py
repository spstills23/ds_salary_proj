# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:10:30 2020

@author: seans
"""
import pandas as pd

df = pd.read_csv('C:/Users/seans/OneDrive/Desktop/Python/ds_salary_proj/glassdoor_jobs.csv')


#salary parsing *
#company name text only
#state field
#parsing through job description  (python, etc)

#SALARY PARSING
#creates and splits salary values where there is a hourly wage into new col
df['hourly']=df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided']=df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)


df = df[df['Salary Estimate'] != '-1']

#splitting the salary from 'estimate' remark and taking out 'K' and '$'
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
min_hr = minus_Kd.apply(lambda x:x.lower().replace('per hour','').replace('employer provided salary:',''))
#making the min and max salary their own col and AVG
df['min_salary'] = min_hr.apply(lambda x:int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x:int(x.split('-')[1]))
df['average_salary'] = (df.min_salary + df.max_salary) / 2


#COMPANY NAME TEXT ONLY
#looks like every company that has a rating also has 3 number characters at the end of their name so split on that
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)

#STATE FIELD
#split state from location
df['job_state'] = df['Location'].apply(lambda x: x.split(",")[1] if len(x.split(",")) == 2 else x.split(",")[0]) 
df.job_state.value_counts()
df['job_state'] = df['job_state'].apply(lambda x: x.replace('New Jersey','NJ').replace('Oregon','OR').replace('Virginia','VA').replace('California','CA').replace('Nebraska','NE'))
df.job_state.value_counts()

#setting col to check if job is at headquarters state
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1 )

# COMPANY AGE
df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2020 - x)

#PARSING JOB DESCRIPTION
#parse out data science tools
#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
#r studio
df['r_studio_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() or ' r ' in x.lower() else 0)
#spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
#aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
#excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

#EXPORT DF TO CSV
df.to_csv('salary_data_cleaned.csv', index=False)



