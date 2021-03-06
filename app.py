from flask import *
import pickle
import pandas as pd
import os
from sklearn import preprocessing
app = Flask(__name__)


@app.route('/')
def main():
    #もしoutputが残っていれば削除
    if(os.path.exists('output.csv')):
        os.remove('output.csv')
    return render_template('submit.html')


@app.route('/output', methods=['POST'])
def do_pred():
    # 送信されたファイルを取得
    send_data = request.files['send_data']
    try:
        test_x = pd.read_csv(send_data)
        idx = test_x['お仕事No.']
    except:
        message = 'csvファイルを送信してください'
        return render_template('error.html', message=message)

    # 勤務開始時刻、勤務終了時刻を計算
    try:
        test_x['start_time'] = test_x['期間・時間\u3000勤務時間'].str[:2].str.strip(':').astype(int)
    except:
        message = 'csvファイルの形式を確認してください(勤務時間のカラムがありません)'
        return render_template('error.html', message=message)
    test_x['end_time'] = test_x['期間・時間\u3000勤務時間'].str.extract('〜(\d{1,2}:\d{2})', expand=False)
    test_x['end_time'] = test_x['end_time'].str[:2]
    test_x['end_time'] = test_x['end_time'].str.strip(':').astype(int)
    test_x['start_time'] = test_x['start_time'].replace({0: 24})
    test_x['end_time'] = test_x['end_time'].replace({0: 24})
    test_x['job_time'] = test_x['end_time'] - test_x['start_time']

    #勤務時間0を24に変更
    test_x['job_time'] = test_x['job_time'].replace({0:24})

    # 予測に使用するカラムを抽出
    try:
        test_x = test_x[['勤務地\u3000最寄駅1（駅名）', '職種コード', '週2・3日OK', '正社員登用あり',
                        '勤務地\u3000都道府県コード', '大量募集', '土日祝休み','駅から徒歩5分以内',
                        '車通勤OK', '未経験OK', '16時前退社OK', '勤務地\u3000市区町村コード','Wordのスキルを活かす',
                        '短時間勤務OK(1日4h以内)', '交通費別途支給', '英語力を活かす', '給与/交通費\u3000給与下限',
                        'フラグオプション選択', '1日7時間以下勤務OK', '派遣スタッフ活躍中', '扶養控除内',
                        '大手企業', 'シフト勤務','経験者優遇','学校・公的機関（官公庁）', '英語力不要',
                        '土日祝のみ勤務','期間・時間\u3000勤務期間','start_time','残業なし','オフィスが禁煙・分煙',
                        'end_time', '給与/交通費\u3000交通費', '残業月20時間未満', '服装自由',
                        'job_time','会社概要\u3000業界コード','Excelのスキルを活かす','平日休みあり']]
    except:
        message = 'csvファイルの形式を確認してください(必要なカラムがありません)'
        return render_template('error.html', message=message)


    # 勤務地最寄駅のカラムをラベルエンコーディングする
    le = pickle.load(open('labelencoder.pkl', 'rb'))
    try:
        test_x['勤務地\u3000最寄駅1（駅名）'] = le.transform(test_x.loc[:,'勤務地\u3000最寄駅1（駅名）'].values)
    except:
        message = 'データに無効な駅名が含まれています'
        return render_template('error.html', message=message)

    #学習済みモデルをロード
    model = pickle.load(open('trained_model.pkl', 'rb'))
    predicted = model.predict(test_x)
    # 予測ファイルを作成
    submit = pd.DataFrame()
    submit['お仕事No.'] = idx
    submit['応募数 合計'] = predicted
    #予測が0未満のときに0にする
    submit['応募数 合計'] = submit['応募数 合計'].apply(lambda x : 0 if x < 0 else x)
    submit.to_csv('output.csv', index=False)
    return render_template('output.html')


@app.route('/output/download', methods=['POST'])
def download():
    return send_file('output.csv',
                     mimetype='csv',
                     attachment_filename='output.csv',
                     as_attachment=True)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)