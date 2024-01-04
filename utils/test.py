import re

res = """根據提供的文件，以下是報告中必須包含的主要信息摘要：

1. 標準的標題：兆豐金控永續報告書 (Cathay Financial Holding Co., Ltd. Sustainability Report)
2. 關鍵信息或基本欄位：
a. 提升員工職能而實施之方案以及提供之協助的類型和範疇 (Programs and measures to enhance employee skills and provide assistance)
b. 提供因退休或終止勞雇關係而結束職涯之員工，以促進繼續就業能力與生涯規劃之過渡協助方案 (Programs and measures to assist employees who have retired or left the company)
3. 例子，以確保每個人都理解每個欄位：
a. 兆豐金控提供了員工培訓計畫，包括單位主管培訓計畫、菁英幹部培訓計畫、海外分行業務講習班等，以提升員工的職能和技能。 (Cathay Financial Holding Co., Ltd. provides employee training programs, including unit manager training, elite talent training, and overseas business training, to enhance employees' skills and abilities.)
b. 兆豐金控為了促進員工的繼續就業能力和生涯規劃，提供了退休或終止勞雇關係而結束職涯之員工的過渡協助方案，包括職業輔導、職業訓練、退休金規劃等。 (Cathay Financial Holding Co., Ltd. provides career guidance, training, and retirement planning services to assist employees who have retired or left the company in their career development and life planning.)

根據以上信息，報告應包含以下章節：

1. 引言
2. 員工培訓和發展
3. 退休和職業轉型協助
4. 結論

報告還應包含相關數據和統計，以支持提供的信息，例如參加培訓計畫的員工數量、受到職業輔導和訓練的員工數量，以及退休或轉型到新職業的員工數量。
"""
if res.strip().endswith("。"):
    print("End of sentence.")