# import modules
import shutil

# Storing Console's Width
var = shutil.get_terminal_size().columns
# Introduction
print("WELCOME TO WAGE CALCULATOR".center(var))
print("This program will calculate your wage from the hours you have worked".center(var))
print("The amount will be stored in the payroll.csv along with start date, stop date and hours worked".center(var))


# Take user input & convert date input to datetime object
def take_input():
    from datetime import datetime
    try:
        enter_client_name = input('Enter your client name: ')
        enter_project_title = input('Enter your project title: ')
        enter_start_date = input('Enter your datetime (2020-07-30 16:20): ')
        enter_end_date = input('Enter your datetime (Leaving field empty will set end time to current time) (2020-07-30 16:20): ')
        end_time = datetime.now()
        format = "%Y-%m-%d %H:%M"
    except ValueError:
        print("Incorrect format")
    start_time = datetime.strptime(enter_start_date, format)
    if len(enter_end_date) == 0:
        end_time = datetime.now()
    else:
        end_time = datetime.strptime(enter_end_date, format)
    return enter_client_name, enter_project_title, start_time, end_time


# Calculate working hours
def working_hours():
    global client_name
    global project_name
    client_name, project_name, start_datetime, end_datetime = take_input()
    working_hours_difference = end_datetime - start_datetime
    work_hours = working_hours_difference.total_seconds() / 60 ** 2
    return work_hours


# Calculate wages
work_hours = working_hours()


def calculate_wages():
    from decimal import Decimal
    # work_hours=working_hours()
    money_per_hour = Decimal('5.0')
    wages_earn = Decimal(work_hours) * money_per_hour
    final_wages = ('%.2f' % wages_earn)
    return final_wages


# Display Wages
wage_earn = calculate_wages()
print(f'You have worked for {work_hours} hours and wages earned is ${wage_earn}')


# Writing to CSV File and also displaying the tracker data
def save_to_csv():
    import pandas as pd
    tracker_data = {'client': [client_name],
                    'project title': [project_name],
                    'work hours': [work_hours],
                    'wage $': [wage_earn]}
    dataset = pd.DataFrame(tracker_data, columns=['client', 'project title', 'work hours', 'wage $'])
    # print(dataset)
    dataset.to_csv('payroll.csv', sep='\t', header=None, mode='a')
    return dataset


print(save_to_csv())
