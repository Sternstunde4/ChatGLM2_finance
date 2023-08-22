import json
import os
import string
import sys

# 官方txt文件输入位置，末尾需要有/
dir_in = "D:\\LLM_dev\\FinGPT-intern\\pdf_to_txt\\test_txt\\"

# 提取的三表输出位置，末尾需要有/
dir_out = "D:\\LLM_dev\\FinGPT-intern\\txt2table\\"

class Tuple:
    def __init__(self, key, value):
        self.key = key
        self.value = value

class Report:
    def __init__(self):
        self.balance = []
        self.profit = []
        self.cashflow = []

class RawRecord:
    def __init__(self):
        self.TType = ""
        self.Inside = ""

class Record:
    def __init__(self):
        self.TType = ""
        self.Inside = ""
        self.InsideList = []

def extract(recordList):
    conditionFlag = 0
    report = Report()
    balance = []
    profit = []
    cashflow = []

    for record in recordList:
        if conditionFlag == 0 and record.TType == "text" and record.Inside.endswith("财务报表") and record.Inside.startswith("二"):
            conditionFlag = 1
        elif conditionFlag == 1 and record.TType == "text" and record.Inside.endswith("合并资产负债表"):
            conditionFlag = 2
        elif conditionFlag == 2 and record.TType == "excel":
            s = record.InsideList
            if len(s) == 3:
                balance.append(Tuple(s[0], s[1]))
            elif len(s) == 4:
                balance.append(Tuple(s[0], s[2]))
        elif conditionFlag == 2 and record.TType == "text" and record.Inside.endswith("母公司资产负债表"):
            conditionFlag = 3
        elif (conditionFlag == 2 or conditionFlag == 3) and record.TType == "text" and record.Inside.endswith("合并利润表"):
            conditionFlag = 4
        elif conditionFlag == 4 and record.TType == "excel":
            s = record.InsideList
            if len(s) == 3:
                profit.append(Tuple(s[0], s[1]))
            elif len(s) == 4:
                profit.append(Tuple(s[0], s[2]))
        elif conditionFlag == 4 and record.TType == "text" and record.Inside.endswith("母公司利润表"):
            conditionFlag = 5
        elif (conditionFlag == 4 or conditionFlag == 5) and record.TType == "text" and record.Inside.endswith("合并现金流量表"):
            conditionFlag = 6
        elif conditionFlag == 6 and record.TType == "excel":
            s = record.InsideList
            if len(s) == 3:
                cashflow.append(Tuple(s[0], s[1]))
            elif len(s) == 4:
                cashflow.append(Tuple(s[0], s[2]))
        elif conditionFlag == 6 and record.TType == "text" and (record.Inside.endswith("母公司现金流量表") or record.Inside.endswith("合并所有者权益变动表")):
            break

    report.balance = balance
    report.cashflow = cashflow
    report.profit = profit
    return report

def get_all_report(filepath):
    filePreName = filepath.split("\\")[-1]
    balanceFileName = dir_out + filePreName + "_balance.txt"
    profitFileName = dir_out + filePreName + "_profit.txt"
    cashflowFileName = dir_out + filePreName + "_cashflow.txt"

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            recordList = []
            id = 0
            for line in file:
                rawRecord = RawRecord()
                record = Record()
                id += 1
                if id == 835:
                    jsonStr = line
                    rawRecord = json.loads(jsonStr)
                    if rawRecord['TType'] == "页眉" or rawRecord['TType'] == "页脚":
                        continue
                    record.TType = rawRecord['TType']
                    s = rawRecord['Inside']
                    if isinstance(s, str):
                        if len(s) == 0:
                            continue
                        record.Inside = s
                    elif isinstance(s, list):
                        insideList = []
                        for item in s:
                            insideList.append(item)
                        record.InsideList = insideList
                    recordList.append(record)

            report = extract(recordList)

            with open(balanceFileName, 'w') as balanceFile:
                for t in report.balance:
                    balanceFile.write(t.key.replace("\n", "") + "\001" + t.value.replace("\n", "") + "\n")

            with open(profitFileName, 'w') as profitFile:
                for t in report.profit:
                    profitFile.write(t.key.replace("\n", "") + "\001" + t.value.replace("\n", "") + "\n")

            with open(cashflowFileName, 'w') as cashflowFile:
                for t in report.cashflow:
                    cashflowFile.write(t.key.replace("\n", "") + "\001" + t.value.replace("\n", "") + "\n")

    except Exception as e:
        print("Error:", e)

def get_all_report_txt():
    dir_path = dir_in

    try:
        with open("./output.txt", 'r', encoding='utf-8') as file:
            txtlines = file.readlines()

            for eachline in txtlines:
                filepath = dir_path + eachline.strip() + ".txt"
                print(filepath)
                get_all_report(filepath)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    get_all_report_txt()