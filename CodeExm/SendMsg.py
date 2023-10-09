import requests

def sendMsg(title="Sent by my program",name='A4000',content="Program Finished!"):

    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                         json={
                             "token": "cc17ddb38d68",
                             "title": title,
                             "name": name,
                             "content": content
                         })
    print(resp.content.decode())
    return "...>>>Token Sent!<<<..."


# from SendMsg import sendMsg
# sendMsg()
