from aip import AipOcr
""" 你的 APPID AK SK """
APP_ID = '17026443'
API_KEY = '9ZOTrTfgmcCCNt15szm725QZ'
SECRET_KEY = 'TU8Ct5Xm0Xrnw8NC7WKBqedzGkT6UzSc'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('C:\\Users\\liuyuntao\\Desktop\\2.jpg')
idCardSide = "back"

""" 调用身份证识别 """
client.idcard(image, idCardSide);

""" 如果有可选参数 """
options = {}
options["detect_direction"] = "true"
options["detect_risk"] = "false"

""" 带参数调用身份证识别 """
print(client.idcard(image, idCardSide, options))



# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()

# image = get_file_content('C:\\Users\\liuyuntao\\Desktop\\1.jpg')

# # """ 调用驾驶证识别 """
# client.drivingLicense(image);

# """ 如果有可选参数 """
# options = {}
# options["detect_direction"] = "true"

# """ 带参数调用驾驶证识别 """
# #client.drivingLicense(image, options)
# print(client.drivingLicense(image, options))