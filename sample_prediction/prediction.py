#prediction code
#load saved model
with open('model_40_5_20.pkl', 'rb') as f:
    gp2 = pickle.load(f)
scaler2 = pickle.load(open('scaler_20_5_20.pkl', 'rb'))
df5=pd.read_csv("C22_2_s12_100 _test.csv", header=None)
data_pred=np.array(df5, dtype=float)
pred_data=scaler2.transform(data_pred)
prediction=gp2.predict(pred_data[:,0:3])
con = np.concatenate((pred_data[:,0:3], prediction), axis=1)
out_data=scaler2.inverse_transform(con)
#out_data
#visulaization of y stress
mysig=out_data[:,4]
sig_y=mysig.reshape(56,10)
plt.imshow(sig_y)
plt.colorbar()
plt.show()
#save data
df_out=pd.DataFrame(out_data)
df_out.to_excel(r'C:\Users\smz3631\Desktop\AL_composite\C22_2_s12_100_pred.xlsx', index = False)
