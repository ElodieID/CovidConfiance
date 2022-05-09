import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import re
import prince
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px  
import plotly.graph_objects as go
#from streamlit.scriptrunner import RerunData
#from streamlit.scriptrunner import RerunException
import sys 
#st.set_page_config(layout="wide")

BACKGROUND_COLOR = 'white'
COLOR = 'black'
max_width = 1500
max_width_100_percent = False
padding_top = 1
padding_right = 10
padding_left = 1
padding_bottom = 10,
color = COLOR
background_color = BACKGROUND_COLOR

max_width_str = f'max-width: {max_width}px;'
st.markdown(
    f'''
    <style>
        .reportview-container .sidebar-content {{
            padding-top: {padding_top}rem;
        }}
        .reportview-container .main .block-container {{
            {max_width_str}
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {color};
            background-color: {background_color};
        }}
    </style>
    ''',
    unsafe_allow_html=True,
)

#set_page_container_style()
@st.cache(allow_output_mutation=True)
def load_data2():
    # Import de la liste des variabels avec leurs définitions
    #df_KeyLab = pd.read_excel("C:\Work\COVID\Key_Label.xlsx", index_col = 0)
    df_KeyLab = pd.read_excel("Key_Label.xlsx", index_col = 0) 
    Vl = dict(zip(df_KeyLab.index, df_KeyLab["Label 1"]))
    return Vl
         
@st.cache(allow_output_mutation=True)
def load_data3():
    data3_AT = pd.read_csv('ndf_AT.csv',sep = ";",index_col=0, encoding='latin1')
    data3_AU = pd.read_csv('ndf_AU.csv',sep = ";",index_col=0, encoding='latin1')
    data3_BR = pd.read_csv('ndf_BR.csv',sep = ";",index_col=0, encoding='latin1')
    data3_DE = pd.read_csv('ndf_DE.csv',sep = ";",index_col=0, encoding='latin1')
    data3_FR = pd.read_csv('ndf_FR.csv',sep = ";",index_col=0, encoding='latin1')
    data3_IT = pd.read_csv('ndf_IT.csv',sep = ";",index_col=0, encoding='latin1')
    data3_NZ = pd.read_csv('ndf_NZ.csv',sep = ";",index_col=0, encoding='latin1')
    data3_PL = pd.read_csv('ndf_PL.csv',sep = ";",index_col=0, encoding='latin1')
    data3_SE = pd.read_csv('ndf_SE.csv',sep = ";",index_col=0, encoding='latin1')
    data3_UK = pd.read_csv('ndf_UK.csv',sep = ";",index_col=0, encoding='latin1')
    data3_US = pd.read_csv('ndf_US.csv',sep = ";",index_col=0, encoding='latin1')
    data3 = pd.concat([data3_AT,data3_AU])
    data3 = pd.concat([data3,data3_BR])
    data3 = pd.concat([data3,data3_DE])
    data3 = pd.concat([data3,data3_FR])
    data3 = pd.concat([data3,data3_IT])
    data3 = pd.concat([data3,data3_NZ])
    data3 = pd.concat([data3,data3_PL])
    data3 = pd.concat([data3,data3_SE])
    data3 = pd.concat([data3,data3_UK])
    data3 = pd.concat([data3,data3_US])
    data3 = data3.replace('',  float("nan"))
    data3 = data3.replace('NA',  float("nan"))
    data3 = data3.replace('(NA)',  float("nan"))
    data3 = data3.replace(99,  float("nan"))
    data3.dropna(how='all', axis=1, inplace=True)
    return data3


@st.cache(allow_output_mutation=True)
def load_data1():
    data1_AT = pd.read_csv('df_AT.csv',sep = ";",index_col=0, encoding='latin1')
    data1_AU = pd.read_csv('df_AU.csv',sep = ";",index_col=0, encoding='latin1')
    data1_BR = pd.read_csv('df_BR.csv',sep = ";",index_col=0, encoding='latin1')
    data1_DE = pd.read_csv('df_DE.csv',sep = ";",index_col=0, encoding='latin1')
    data1_FR = pd.read_csv('df_FR.csv',sep = ";",index_col=0, encoding='latin1')
    data1_IT = pd.read_csv('df_IT.csv',sep = ";",index_col=0, encoding='latin1')
    data1_NZ = pd.read_csv('df_NZ.csv',sep = ";",index_col=0, encoding='latin1')
    data1_PL = pd.read_csv('df_PL.csv',sep = ";",index_col=0, encoding='latin1')
    data1_SE = pd.read_csv('df_SE.csv',sep = ";",index_col=0, encoding='latin1')
    data1_UK = pd.read_csv('df_UK.csv',sep = ";",index_col=0, encoding='latin1')
    data1_US = pd.read_csv('df_US.csv',sep = ";",index_col=0, encoding='latin1')
    data1 = pd.concat([data1_AT,data1_AU])
    data1 = pd.concat([data1,data1_BR])
    data1 = pd.concat([data1,data1_DE])
    data1 = pd.concat([data1,data1_FR])
    data1 = pd.concat([data1,data1_IT])
    data1 = pd.concat([data1,data1_NZ])
    data1 = pd.concat([data1,data1_PL])
    data1 = pd.concat([data1,data1_SE])
    data1 = pd.concat([data1,data1_UK])
    data1 = pd.concat([data1,data1_US])
    data1.country = data1.country.astype(object) 
    data1.wave = data1.wave.astype(object)
    data1 = data1.replace("I don't know",  float("nan"))
    data1 = data1.replace('',  float("nan"))
    data1 = data1.replace('NA',  float("nan"))
    data1 = data1.replace('(NA)',  float("nan"))
    data1 = data1.replace(99,  float("nan"))
    data1.dropna(how='all', axis=1, inplace=True) 
    return data1


Vl = load_data2()

df_Num = load_data3()
#Col_Select = df_Num.columns
df_O = load_data1()  



#st.set_page_config(layout="wide")
#st.set_page_config(layout="centeride")
# Fonciton retournant la liste des variables à valeur unique
def Var_Uni(df_):
    Var_Sel= []
    for Labl in df_.columns:
      if df_[Labl].value_counts().shape[0]==1 : 
          Var_Sel.append (Labl) 
    return Var_Sel

# Fonciton retournant la liste des variables avec autant de modalité différente que de valeurs
def Var_inv_Uni(df_):
    Var_Sel= []
    for Labl in df_.columns:
        if (df_[Labl].value_counts().shape[0]==df_[Labl].shape[0]) & (df_[Labl].dtype == 'object'): 
             Var_Sel.append (Labl)   
    return Var_Sel  


# identification des Var ayant du texte et des nombres.
def IsNumTxt_In_List(lst):
    Num = False
    Txt = False
    lst2 = [x for x in lst if pd.isnull(x) == False and x != 'nan']
    for i in lst2:
        if isinstance(i, (int, float)) or i.isdigit():
            Num = True
        else: 
            Txt = True 
    if Num and Txt: return 'Num_Txt'
    elif Num == True and Txt == False: return 'Num'
    elif Txt == True and Num == False: return 'Txt'
    else: return ''

   
#Suppression des lignes qui contiennent au moins une valeur numerique   
def drop_Num(df_):
    for Col in df_.columns:
        df_[Col] = df_[Col].apply(lambda x: np.nan if isinstance(x, (int, float)) else x)
        return df_.dropna()
    

#Renvoi le nom de la variable suivant selon sa racine
def Uniq_Var(Txt,Vars):
    for var in Vars:
        if var.endswith(Txt):
            return var   
    
def add_value_label(x_list,y_list,txt_list,annotate,width):
    for i in range(0, len(x_list)):
        print (x_list[i])
        if annotate == 0: plt.text(i,y_list[i]+0.07,txt_list[i],ha='center',style='italic')
        if annotate == 1: plt.text(i-width/2,y_list[i]+0.07,txt_list[i],ha='left',style='italic',rotation = 70,fontsize = 8)
        if annotate == 2: plt.text(i+width/2,y_list[i]+0.07,txt_list[i],ha='left',style='italic',rotation = 70,fontsize = 8)    
        if annotate == 3: plt.text(x_list[i],y_list[i]+0.06,txt_list[i],ha='center',style='italic',rotation = 90,fontsize = 8)    
        
def Title_Format(x):
    Title_0 =''
    Title_1 =''
    for mot in x.split():
        if len (Title_0) < 50:
            Title_0 = Title_0 + mot + ' ' 
        else:
            Title_1 = Title_1 + Title_0 + mot + '\n'
            Title_0 =''         
    Title_1 = Title_1 + Title_0
    return Title_1
        
    

#********************************
from PIL import Image 
img = Image.open("covid.jpg") 
st.sidebar.image(img)
st.sidebar.title("Covid & Confiance")

pages = ["Introduction","Visualisation", "Modélisation"]
page = st.sidebar.radio('', pages)

st.markdown(
       """
       <style>
       .main {
       background-color : #e3e3e3;
       }
       </style>
       """, unsafe_allow_html=True
       )
   
background_color = '#e3e3e3'







if page==pages[0]:
          
    st.image(img, use_column_width='always')
    
    st.title("Introduction")
    
    
    st.markdown('<div style="text-align: justify;">Le projet Citizens Attitudes Under COVID-19 Pandemic dit CAUCP a pour objectif d’étudier l’évolution de l''opinion publique pendant la pandémie de Covid19 dans 11 démocraties entre mars et décembre 2020. Plus particulièrement, d’étudier la confiance des habitants en leur gouvernement et en la science.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div style="text-align: justify;">Ce projet est coordonné par Sylvain Brouard (Sciences Po, CEVIPOF), Michael Becher (IAST, Université de Toulouse 1), Martial Foucault (Sciences Po, CEVIPOF) et Pavlos Vasilopoulos (University de York and CEVIPOF).</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div style="text-align: justify;">Il est constitué d’un ensemble de questionnaires réalisés durant les différentes vagues de Covid19. Chaque questionnaire contient entre 200 et 300 questions qui vont permettent d''étudier les conséquences politiques, comportementales, et attitudinales de la crise dans les différents contextes, et à travers le temps.</div>', unsafe_allow_html=True)
    
    df = df_O.copy(deep=True)
    df.rename(columns={'AGE': 'Age', 'panel': 'Nb participants'}, inplace=True)
   
    st.write("")
    
    st.header("Description rapide des données")
    
    st.subheader("Nombre de personnes interrogées par pays et par vague :")
    
    st.write("")
    
    df_1 = pd.crosstab(df.country, df.wave)
     
    fig0= px.bar(df_1,barmode='group', color='wave',color_discrete_map=
                 {'1.0':'grey',
                  '2.0':'gold',
                  '3.0':'darkred',
                  '4.0':'black'})   
    
    fig0.update_layout(
        width=800,
        height=500,
        margin = dict(l=1,r=1,b=1,t=1),
        font=dict(color='#383635', size=15),
        paper_bgcolor=background_color)    
    
    fig0.update_xaxes(title_text='')   
    fig0.update_yaxes(title_text='Nb participants')
    
    st.write(fig0)
    
    st.subheader("Catégorie d'âge ayant le plus répondu :")
   
    st.write("") 
            
    #Age
     
    df_2_temp= df.groupby(by=df.Age).count().reset_index()
    df_temp= df_2_temp.sort_values(by = 'Nb participants',ascending = False).head(50)
         
    mycolumns = ['Age','Nb participants']    
    top_n = st.slider("Combien de catégories d'âge souhaitez vous afficher?", min_value=2, max_value=10, value=3, step=1)
    top_n=int(top_n)
    
    df_3 = df_temp[mycolumns].sort_values(by = 'Nb participants',ascending = False).head(top_n)
       
    fig1 = go.Figure(data=go.Table(
        columnwidth=[1,2],
        header=dict(values=list(df_3.columns),
                fill_color= '#FFCE33',
                line_color="#7F1D18",
                align='center'),
        cells=dict(values=[df_3.Age, df_3['Nb participants']],
                fill_color='#e3e3e3',
                line_color="#7F1D18",
                height=30,
                align=['center', 'center'])))
    
    fig1.update_layout(margin=dict(l=5,r=5,b=10,t=10),
                      paper_bgcolor = background_color,
                      width=800,
                      height=350,
                      font=dict(size=20))
         
    st.write(fig1)
       
      #SEXE 
       
    col1,colt, col2 = st.columns((3,1,4))

    with col1:
        st.subheader("Répartition Homme/Femme :")  
        
        mycolumns1 = ['SEXE','Nb participants'] 
        df_1_temp= df.groupby(by=df.SEXE).count().reset_index()
        df_sexe = df_1_temp[mycolumns1].sort_values(by = 'Nb participants',ascending = False)

              
        fig2= px.pie(df_sexe, values= 'Nb participants',names ='SEXE', 
                     color='SEXE',color_discrete_map={'Male':'darkred',
                            'Female':'gold'})
        fig2.update_layout(showlegend = False,
                          width = 300,
                          height = 300,
                          margin=dict(l=1,r=1,b=1,t=1),
                          font=dict(color='#383635', size=25),
                          paper_bgcolor=background_color,
                          )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        
        st.write(fig2)

    with col2:
        st.subheader("Nb de personnes indiquant ayant eu le COVID :")
        
        st.write("")
     
        df_cov=df[df['COVID20']=='Yes']
        mycolumns_cov = ['SEXE','id'] 
        
        df_cov_temp= df_cov.groupby(by=df_cov.SEXE).count().reset_index()
        df_covid = df_cov_temp[mycolumns_cov].sort_values(by = 'id',ascending = False)
        
        fig4= px.bar(df_covid,barmode='group', color='SEXE',color_discrete_map=
                  {'Male':'darkred',
                  'Female':'gold',
                  })   
    
        fig4.update_layout(
            width=600,
            height=400,
            margin = dict(l=1,r=1,b=1,t=1),
            font=dict(color='#383635', size=15),
            paper_bgcolor=background_color)    
        
        fig4.update_xaxes(title_text='')   
        fig4.update_yaxes(title_text='')
    
        st.write(fig4)
    
    #Vaccin
    st.subheader("Si un vaccin été disponible dans les prochains mois, vous feriez vous vacciner ?")
    st.text("Réponses uniquement pour le questionnaire de la vague 4")
    
    df_vacc=df[df['VACC1']!= 'Nan']
    mycolumns_vac = ['VACC1','id'] 

    df_vac_temp= df_vacc.groupby(by=df_vacc.VACC1).count().reset_index()
    df_vaccin = df_vac_temp[mycolumns_vac].sort_values(by = 'VACC1')
    
    fig5 = px.line(df_vaccin, x="VACC1", y="id", title='')
    
    fig5.update_layout(
        width=800,
        height=400,
        margin = dict(l=1,r=1,b=1,t=1),
        font=dict(color='#383635', size=15),
        paper_bgcolor=background_color,
        )    

    fig5.update_xaxes(title_text='')   
    fig5.update_yaxes(title_text='')
    
    fig5.update_traces(line_color="#000000")
    
    st.write(fig5)
    
elif page==pages[1]:
        def Default_Value(x_bool):
           Bobtemp = True 
           #st.session_state.Def = x_bool
           #if x == False:st.session_state.lab =  O_Feats
           
        st.sidebar.title("Visualisation") 

        df_O = df_O.loc[:,Vl.keys()]
        
        labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]
        df_O.AGE = pd.cut(df_O.AGE, np.arange(0, 101, 10),
               include_lowest=False, right=False,
               labels=labels)
        
        df_O.AGE = df_O.AGE.astype(str)
        df_M = df_O.copy(deep=True)
        
        O_Country_TRI = [int(str(i)[:2].replace(".","")) for i in list(df_O.country.value_counts().index)]
        O_Country_To_Select = pd.DataFrame(df_O.country.value_counts().index,O_Country_TRI).sort_index().iloc[:,0]
        O_Country = st.sidebar.multiselect('Country name',(O_Country_To_Select),on_change=Default_Value(True))
        
        if len(O_Country) == 0: 
            st.stop()
            sys.exit()
        df_O_Temp = df_O.loc[df_M.country == O_Country[0]].dropna(how='all', axis=1)
        if len(O_Country) > 1: 
            for Country_ in O_Country[1:]:
                df_O_Temp = pd.concat([df_O_Temp,df_O.loc[df_M.country == Country_].dropna(how='all', axis=1)],join ='inner') 
    
        df_O = df_O_Temp.dropna(how='all', axis=1)  
        
        
        st.sidebar.write('Wave of the survey')
        W1 = st.sidebar.checkbox('Wave 1 (Mars 2020)',on_change=Default_Value(True))
        W2 = st.sidebar.checkbox('Wave 2 (Avril 2020)',on_change=Default_Value(True))
        W3 = st.sidebar.checkbox('Wave 3 (Juin 2020)',on_change=Default_Value(True))
        W4 = st.sidebar.checkbox('Wave 4 (Décembre 2020)',on_change=Default_Value(True))
        O_Wave_Selected =[]
        if W1: O_Wave_Selected.append(1.0)
        if W2: O_Wave_Selected.append(2.0)
        if W3: O_Wave_Selected.append(3.0)
        if W4: O_Wave_Selected.append(4.0)   
        
        if len(O_Wave_Selected) == 0: 
            st.stop()
            sys.exit()
        
        df_O = df_O.loc[df_O.wave.isin(O_Wave_Selected)]
        df_O.dropna(how='all', axis=1, inplace=True) 
        
        dic_df_O ={}
        for W in O_Wave_Selected:
            df_Temp = df_O.loc[df_O.wave == W]
            dic_df_O[W] = df_Temp
    
        df_O_Temp = dic_df_O[O_Wave_Selected[0]].dropna(how='all', axis=1)
        if len(O_Wave_Selected) > 1: 
            for W in O_Wave_Selected[1:]:
              df_O_Temp = pd.concat([df_O_Temp,df_O.loc[df_O.wave == W].dropna(how='all', axis=1)],join ='inner')  
        df_O = df_O_Temp.dropna(how='all', axis=1) 
        
        st.sidebar.write('How much do you trust…?:')
        B4_0 = st.sidebar.checkbox('The mayor of your town/city',disabled=not 'B4_0' in df_O.columns,on_change=Default_Value(True))
        B4_6 = st.sidebar.checkbox('The government',disabled=not 'B4_6' in df_O.columns,on_change=Default_Value(True))
        B4_7 = st.sidebar.checkbox('Scientists',disabled=not 'B4_7' in df_O.columns,on_change=Default_Value(True))
        B4_8 = st.sidebar.checkbox('Doctors',disabled=not 'B4_8' in df_O.columns,on_change=Default_Value(True))
        B4_10 = st.sidebar.checkbox('Big companies',disabled=not 'B4_10' in df_O.columns,on_change=Default_Value(True))
        B4_12 = st.sidebar.checkbox('Journalists',disabled=not 'B4_12' in df_O.columns,on_change=Default_Value(True))
        
        O_Conf_Selected =[]
        if B4_0: O_Conf_Selected.append('B4_0')
        if B4_6: O_Conf_Selected.append('B4_6')
        if B4_7: O_Conf_Selected.append('B4_7')   
        if B4_8: O_Conf_Selected.append('B4_8')
        if B4_10: O_Conf_Selected.append('B4_10')
        if B4_12: O_Conf_Selected.append('B4_12')
        
        Lst_Confi_Key = []
        Lst_Confi_val =[]
        for key,val in Vl.items():
            if key in O_Conf_Selected:
                Lst_Confi_Key.append(key)
                Lst_Confi_val.append(val) 
        Lst_Vl_Val=[]
        for Col in df_O.columns:
            Lst_Vl_Val.append(Vl[Col])          
        Lst_Vl_Val=list(set(Lst_Vl_Val))
        Lst_Vl_Val.sort()
        List_Feats = [x for x in Lst_Vl_Val if not x in list(Lst_Confi_val)]
        
        col1,col2 = st.columns((12,1))
        
        with col1:        
            CheckDef = st.checkbox('Lock variable')
            
        with col1:    
            if CheckDef == True : 
                if 'lab' in st.session_state:
                    #O_Feats = st.selectbox('Variables of interest',List_Feats,index= List_Feats.index(st.session_state.lab))
                    O_Feats = st.session_state.lab
                    #st.session_state.lab =  O_Feats
                else:
                    O_Feats = st.selectbox('Variables of interest',List_Feats) 
                    st.session_state.lab =  O_Feats
            else: 
                O_Feats = st.selectbox('Variables of interest',List_Feats)
                st.session_state.lab =  O_Feats        
        
        if 'wave' in O_Feats: 
            O_Feats = None
            Abs_wave = True
        else:
            Abs_wave = False
            
        Lst_Feats_Key =[]
        Lst_Feats_val =[]
        for key,val in Vl.items():
            if val == O_Feats:
                Lst_Feats_Key.append(key)
                Lst_Feats_val.append(val)
                
        df_O = df_O.loc[:,['wave','country'] + Lst_Feats_Key + Lst_Confi_Key]
        df_O.dropna(how='all', axis=1, inplace=True) 
        
        replace = ['not at all','not a lot','somewhat','completely',"Don't trust at all","Don't trust a lot","Trust somewhat","Trust completely"]
        to_replace = [1,2,3,4,1,2,3,4]
        df_O[Lst_Confi_Key] = df_O[Lst_Confi_Key].replace(replace,to_replace)
        
        st.sidebar.write('Paramètres graphiques')
        Leg = st.sidebar.checkbox("Légende")
        
        
        List_Labl_Txt = []
        for Labl in df_O[Lst_Feats_Key]: 
            if IsNumTxt_In_List(list(df_O[Labl]))=='Txt'and Labl != 'AGE': 
                Bob_Temp  = pd.concat([df_Num[Labl].fillna('').astype(str),df_O[Labl]], axis=1,join = 'inner')                
                df_O[Labl] = Bob_Temp.loc[Bob_Temp.isnull().any(axis=1)==False].agg('_-'.join, axis=1)
                List_Labl_Txt.append(Labl)        
                #st.write('TRI', df_O[Labl])
        
        O_Wave_Selected = [str(x) for x in O_Wave_Selected]            
        Vague = "Wave " +  ", ".join(O_Wave_Selected)
        
        from collections import OrderedDict    
        linestyles_dict = OrderedDict(
            [('solid',               (0,())),
             ('loosely dotted',      (0, (1, 3))),
             ('dotted',              (0, (1, 2))),
             ('densely dotted',      (0, (1, 1))),
        
             ('loosely dashed',      (0, (5, 10))),
             ('dashed',              (0, (5, 5))),
             ('densely dashed',      (0, (5, 1))),
        
             ('loosely dashdotted',  (0, (3, 10, 1, 10))),
             ('dashdotted',          (0, (3, 5, 1, 5))),
             ('densely dashdotted',  (0, (3, 1, 1, 1))),
        
             ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
             ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
             ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
        
        c = 0
        l = 0
        Color_Blue = ['#d8e6f2','#b1cee6','#8bb5d9','#649dcd','#3e85c0','#316a9a','#254f73','#18354d']
        Color_Orange = ['#feedcc','#ffdb99','#ffc966','#ffb733','#ffa500','#cc8400','#996300','#664200']
        Color_Blue = ['#ccccfe','#99f','#66f','#33f','#00f','#00c','#009','#006']
        Color = {'B4_0':'green','B4_6':'blue','B4_7':'orange','B4_8':'cyan','B4_10':'magenta','B4_12':'black'}#,'black'}
        linestyle=['solid','densely dotted','dotted','loosely dotted']
        width = 0.1+0.4/len(O_Wave_Selected)
        hatch=[None,".....","...","."]
        Color_Bar=['red','none','none','none']
        edgecolor = [None,'red','red','red']
        
        if len(O_Country) > 0 and len(O_Wave_Selected) > 0 and len(O_Conf_Selected) > 0:        
            if Abs_wave == False:
                for Country_ in O_Country:
                    df_country = df_O.loc[df_O.country == Country_]
                    for label, Title in Vl.items():
                        if label in Lst_Feats_Key:
                            Temp_GS = df_country[label].value_counts().to_frame(name='Effectif_Total')  
                            #st.write('gsgffsgs',df_country[label]) 
                            for Wave_ in O_Wave_Selected:
                                #print(Wave_)
                                
                                Temp_GS = Temp_GS.join(df_country.loc[df_country.wave.astype(float) == float(Wave_)][label].value_counts(normalize = True).to_frame(name='Effectif_%_' +Wave_))
                                Temp_GS = Temp_GS.join(df_country.loc[df_country.wave.astype(float) == float(Wave_)][label].value_counts().to_frame(name='Effectif_'+ Wave_))
                                Temp_GS = Temp_GS.join(df_country.loc[df_country.wave.astype(float) == float(Wave_)].groupby(label).agg({Lst_Confi_Key[0]:'mean'})) 
                                Entete = list(Temp_GS.columns[:-1]) 
                                Entete.append(Lst_Confi_Key[0] + "_wave_" + Wave_)
                                Temp_GS.set_axis(Entete,  axis='columns', inplace=True)
                                if len(Lst_Confi_Key) > 1:
                                    for conf in Lst_Confi_Key[1:]:                 
                                        Temp_GS = Temp_GS.join(df_country.loc[df_country.wave.astype(float) == float(Wave_)].groupby(label).agg({conf:['mean']}))
                                        Entete = list(Temp_GS.columns[:-1]) 
                                        Entete.append(conf + "_wave_" + Wave_)
                                        Temp_GS.set_axis(Entete,  axis='columns', inplace=True)
                            
                                               
    
                            Temp_GS = Temp_GS.sort_index()
                            #st.write('gsgffsddgs',Temp_GS) 
                            #Temp_GS = Temp_GS.dropna(how='any')
                            if label in List_Labl_Txt:
                                
                                Temp_GS["TRI"] = Temp_GS.index
                                #st.write('TRI',Temp_GS)
                                def TRI_GS(x):
                                    if str(x[0]).isdigit():
                                        return int(x[:2].replace('.',''))
                                    else:
                                        return 0                                    
                                Temp_GS["TRI"] =Temp_GS["TRI"].apply(lambda x: TRI_GS(x))
                                Temp_GS.sort_values(by="TRI",inplace = True)
                                
                                Temp_GS = Temp_GS.rename_axis('index').reset_index()
                                Temp_GS['index'] = Temp_GS['index'].transform(lambda x: x[x.find('_-')+2:])
                                Temp_GS.set_index('index',inplace=True)
                                #if len(Temp_GS.index) > 7:
                                #    Temp_GS = Temp_GS.sort_values(by=[Lst_Confi_Key[-1]+ "_wave_" + Wave_])
                            #st.write('gsg',Temp_GS)        
                            fig = plt.figure(figsize=(10,10))
                            x = Temp_GS.index.astype(str) 
                            x_ = np.arange(len(x))   
                            c = 0
                            for conf in Lst_Confi_Key:
                                w_alpha = 1
                                w_line = len(O_Wave_Selected)-1
                                for Wave_ in O_Wave_Selected:                                                     
                                    y = Temp_GS[conf + "_wave_" + Wave_]                       
                                    plt.plot(x,y,label ="Wave " + Wave_+ '; ' + Vl[conf].replace('B4. How much do you trust…? : ','') ,color=Color[conf],linestyle=linestyles_dict[linestyle[w_line]],alpha=0.2+0.8*(w_alpha/int(len(O_Wave_Selected))))
                                    w_line += -1
                                    w_alpha +=1
                                c +=1    
                            
                            i_Wav = 0
                            w_Style = len(O_Wave_Selected)-1
                         
                              
                            for Wave_ in O_Wave_Selected:  
                                y = Temp_GS['Effectif_%_' +Wave_] 
                                plt.bar(x_ + i_Wav* width ,y,width = width - width/10,label ="Wave " + Wave_+ '; ' + 'Effectif',hatch=hatch[w_Style],color = Color_Bar[w_Style], edgecolor =edgecolor[w_Style],lw = 0.1)
                                y_ = Temp_GS['Effectif_'+ Wave_]                                              
                                for i in range(0, len(x_)):
                                    plt.text(i + i_Wav* width ,y[i]+0.07,y_[i],ha='center',style='italic',rotation = 70,fontsize = 8)
                                i_Wav += 1
                                w_Style +=-1
                            
                            font_title = {'family':'serif','color':'black','size':20,'style':'italic'} 
                            font_XY_Labels = {'family':'serif','color':'darkred','size':16}    
                            plt.ylim(0,4)
                            ax = plt.gca()
                            ax.set_axisbelow(True)
                            ax.yaxis.grid(which="both",  linewidth=0.2)
                            ax.xaxis.grid(which="major", linestyle='-', linewidth=0.2)
                            LabelX = Title_Format(Title)
                            plt.xlabel(LabelX, fontdict = font_XY_Labels)
                            plt.ylabel("Confidence index (mean)", fontdict = font_XY_Labels)
                            plt.xticks(fontsize=12,rotation=70,ha="right")
                            plt.yticks(fontsize=12)  
                            Country_Name = re.sub(r'[0-9\.]+', '', Country_).lstrip() + '                                    '
                            Country_Name = Country_Name[:25]
                            plt.title(Country_Name +' '+ u'B4. What confidence do you place?\n', fontdict = font_title, loc = 'right')                
                            plt.axhline(color ="black", linestyle ="-",linewidth=1,y=1)
                            plt.axhline(color ="gray", linestyle ="--",linewidth=0.55,y=2.5)
        
                            if Leg: plt.legend(prop={'size': 12}) 
                            handles, labels = ax.get_legend_handles_labels()    
                                     
                            my_expander = st.expander(Country_Name + ' (' + label + ')', expanded=True)
                            with my_expander:
                                st.pyplot(fig)
        
        
                            
            else:
                df_O.wave =  df_O.wave.apply(lambda x: 'Wave ' + str(x))
                for Conf_ in Lst_Confi_Key:
                    for Wave_ in O_Wave_Selected:
                        Temp_GS = df_O.wave.value_counts().to_frame(name='Effectif_Total')
                        for  Country_ in O_Country:   
                            Temp_GS = Temp_GS.join(df_O.loc[df_O.country == Country_].wave.value_counts(normalize = True).to_frame(name='Effectif_%_' +Country_))
                            Temp_GS = Temp_GS.join(df_O.loc[df_O.country == Country_].wave.value_counts().to_frame(name='Effectif_'+ Country_))
                            Temp_GS = Temp_GS.join(df_O.loc[df_O.country == Country_].groupby('wave').agg({Conf_:'mean'})) 
                            Entete = list(Temp_GS.columns[:-1]) 
                            Entete.append(Conf_+ "_" + Country_)
                            Temp_GS.set_axis(Entete,  axis='columns', inplace=True)                
                    
                    Temp_GS = Temp_GS.sort_index()
                    fig = plt.figure(figsize=(10,10))
                    x = Temp_GS.index.astype(str)                       
                    for Country_ in O_Country: 
                        y = Temp_GS[Conf_+ "_" + Country_]
                        plt.plot(x,y,label = re.sub(r'[0-9\.]+', '', Country_).lstrip())   
                    
                    plt.ylim(1,4)    
                    font_title = {'family':'serif','color':'black','size':20,'style':'italic'} 
                    font_XY_Labels = {'family':'serif','color':'darkred','size':16}    
                    ax = plt.gca()
                    ax.set_axisbelow(True)
                    ax.yaxis.grid(which="both",  linewidth=0.2)
                    ax.xaxis.grid(which="major", linestyle='-', linewidth=0.2)
                    plt.xlabel(Vl['wave'], fontdict = font_XY_Labels)
                    plt.ylabel("Confidence index (mean)", fontdict = font_XY_Labels)
                    plt.xticks(fontsize=12,rotation=70,ha="right")
                    plt.yticks(fontsize=12)                
                    plt.title(Vl[Conf_] + '\n', fontdict = font_title, loc = 'right') 
                    plt.axhline(color ="black", linestyle ="-",linewidth=1,y=1)
                    plt.axhline(color ="gray", linestyle ="--",linewidth=0.75,y=2.5)
            
                    if Leg: plt.legend(prop={'size': 12})  
                    my_expander = st.expander(Lst_Confi_val[Lst_Confi_Key.index(Conf_)], expanded=True)
                    with my_expander:
                        st.pyplot(fig)

                
                
elif page==pages[2]:
    st.title("Modélisation")
 
    st.text("Ceci concerne la partie Machine Learning")
    
    st.line_chart({"data": [1, 5, 2, 6, 2, 1]})

    with st.expander("See explanation"):
     st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")
    
    

   
