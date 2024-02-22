


with open('書き込みたいファイルへのパス', 'r') as f:#書き込みたいファイル
    ja = f.read()

with open('書き込まれるフェイルへのパス', mode='a') as f:#書き込まれるふぁいる
    f.write(ja)








