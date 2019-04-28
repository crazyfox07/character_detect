# @Time    : 2019/4/9 15:25
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : tmp.py
import requests
import urllib
import base64
import datetime


sess = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
}

user_name = 'LCBJ10052'
pppppp = 'TmpnM1FHRnpaR1pu'
workdate = datetime.datetime.now().strftime('%Y-%m-%d')
worktime = '10'
memo = '开发'


# 登录获取cookie
def login():
    login_url = r'http://10.7.13.26:9080/times/login/validate'
    aa = base64.b64decode(base64.b64decode(pppppp.encode('utf8')))
    aa = str(aa, encoding='utf8')
    form_data = {
        'userName': user_name,
        'password':aa
    }
    sess.headers = headers
    sess.post(login_url, data=form_data)
    return sess


# 新增工时
def gongshi_add():
    sess = login()
    param_dict = dict(userid=user_name,
                      workdate=workdate,
                      firstclass='CDT工时',
                      projectid='680',
                      projectleader='褚福州',
                      worktime=worktime,
                      memo=memo)
    param_dict_encode = urllib.parse.urlencode(param_dict, encoding='utf8')
    url = r'http://10.7.13.26:9080/times/times/adddetail?%s' % (param_dict_encode,)
    res = sess.post(url)
    print(res.status_code, res.json())


if __name__ == '__main__':
    gongshi_add()
