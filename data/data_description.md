# Credit Default Dataset Description

## Source
UCI Machine Learning Repository - Default of Credit Card Clients Dataset

## Dataset Characteristics
- **Instances:** 30,000
- **Features:** 25 (24 input + 1 target)
- **Class Distribution:** 77.9% non-defaults, 22.1% defaults

## Feature Categories

### Demographics
- SEX: Gender (1=male, 2=female)
- EDUCATION: Education level 
- MARRIAGE: Marital status
- AGE: Age in years

### Financial
- LIMIT_BAL: Credit limit (NT dollars)
- BILL_AMT1-6: Bill amounts for 6 months
- PAY_AMT1-6: Payment amounts for 6 months

### Target Variable
- default.payment.next.month: Default payment (1=yes, 0=no)
