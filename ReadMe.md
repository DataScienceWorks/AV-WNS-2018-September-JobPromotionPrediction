# AV WNS - Job Promotion Prediction

**Hiring Hackathon : Job Promotion Prediction** 

Website: https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/

**Notes to self**
This is my 4nd hackathon in AV. 


## Tags

Classification 


## Description

## Problem Statement

Your client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:

1. They first identify a set of employees based on recommendations/ past performance
2. Selected employees go through the separate training and evaluation program for each vertical. These programs are based on the required skill of each vertical
3. At the end of the program, based on various factors such as training performance, KPI completion (only employees with KPIs completed greater than 60% are considered) etc., employee gets promotion

For above mentioned process, the final promotions are only announced after the evaluation and this leads to delay in transition to their new roles. Hence, company needs your help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle.

![img](images/wns_hack_im_1.jpg)

They have provided multiple attributes around Employee's past and current performance along with demographics. Now, The task is to predict whether a potential promotee at checkpoint in the test set will be promoted or not after the evaluation process.

------

## Data

| **Variable**         | **Definition**                                               |
| -------------------- | ------------------------------------------------------------ |
| employee_id          | Unique ID for employee                                       |
| department           | Department of employee                                       |
| region               | Region of employment (unordered)                             |
| education            | Education Level                                              |
| gender               | Gender of Employee                                           |
| recruitment_channel  | Channel of recruitment for employee                          |
| no_of_trainings      | no of other trainings completed in previous year on soft skills, technical skills etc. |
| age                  | Age of Employee                                              |
| previous_year_rating | Employee Rating for the previous year                        |
| length_of_service    | Length of service in years                                   |
| KPIs_met >80%        | if Percent of KPIs(Key performance Indicators) >80% then 1 else 0 |
| awards_won?          | if awards won during previous year then 1 else 0             |
| avg_training_score   | Average score in current training evaluations                |
| is_promoted          | (Target) Recommended for promotion                           |

### EVALUATION METRIC

The evaluation metric for this competition is F1 Score.



### PUBLIC AND PRIVATE SPLIT

Test data is further randomly divided into Public (40%) and Private (60%) data.

- Your initial responses will be checked and scored on the Public data.
- The final rankings would be based on your private score which will be published once the competition is over.



### WNS Hackathon Rules

1. Entries submitted after the contest is closed will not be considered.
2. Use of external dataset is strictly prohibited.
3. Use of employee_id as a feature or in any way to build the model is not allowed



### Solution Checker

1. You are only allowed to make 15 submissions per day
2. Adding comments is mandatory for the use of solution checker
3. Comments will help you to refer to a particular solution at a later point in time.



### Final Submission

1. Setting final submission is mandatory. Without a final submission, your entry will not be considered.
2. Code file is mandatory while sending final submission. For GUI based tools, please upload a zip file of snapshots of steps taken by you, else upload code file.
3. The code file uploaded should be pertaining to your final submission.