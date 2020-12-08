import pandas as pd

def get_data():
    data1 = pd.read_table('student-mat.csv', sep = ';')
    data2 = pd.read_table('student-por.csv', sep = ';')

    data = pd.concat([data1,data2])
    data = data[['sex','age','Pstatus', 'higher', 'internet', 'romantic', 'famrel', 'goout', 'Dalc', 'famsize',
    'absences']]

    data['absencebin'] = np.where(data['absences'] > 0, 'Absent Logged', 'Perfect Attendance')

    return data