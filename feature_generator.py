import pandas as pd

def GetCategoricalVariables(maxCategories, dfTrain, dfTest):

    originalVariables = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7'
                     , 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2'
                     , 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7'
                     , 'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4' 
                     , 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2'                     
                     , 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7' 
                     , 'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12'
                     , 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18'
                     , 'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23' 
                     , 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29' 
                     , 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35' 
                     , 'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39', 'Medical_History_40'
                     , 'Medical_History_41']

    categoricalVariables = list()
    for var in originalVariables:

        #TODO: what to do about variables with a crap ton of categories
        if len(dfTrain[var].unique()) > maxCategories:
            print var
            continue

        uniqueValues = dfTrain[var].unique()
        for j in range(len(uniqueValues)):
            attributeName = 'OneHot%s_%s' % (j, var)
            dfTrain[attributeName] = dfTrain[var].apply(lambda x: x == uniqueValues[j])
            dfTest[attributeName] = dfTest[var].apply(lambda x: x == uniqueValues[j])
            categoricalVariables.append(attributeName)
            
    return categoricalVariables

def GetFeatures(dfTrain, dfTest):

    dummyVariables = list()
    dummyVariables += ['Medical_Keyword_' + str(i) for i in range(1, 49)]

    originalVariables = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5']
    continuousVariables = list()
    for var in originalVariables:
        if dfTrain[var].isnull().sum():
	    continuousVar = var + "_NO_NULLS"
	    continuousVariables.append(continuousVar)
	    dummyVariable = var + "_IS_NULL"
	    dummyVariables.append(dummyVariable)
	    dfTrain[dummyVariable] = dfTrain[var].isnull()
	    dfTest[dummyVariable] = dfTest[var].isnull()
	    
	    nonNullTrain = dfTrain[dfTrain[var].notnull()][var]
	    nonNullTest = dfTest[dfTest[var].notnull()][var]
	    mle = pd.concat([nonNullTrain, nonNullTest]).median()
	
	    dfTrain.loc[dfTrain[var].isnull(), continuousVar] = mle
	    dfTrain.loc[dfTrain[var].notnull(), continuousVar] = nonNullTrain
	    dfTest.loc[dfTest[var].isnull(), continuousVar] = mle
	    dfTest.loc[dfTest[var].notnull(), continuousVar] = nonNullTest
	else:
            continuousVariables.append(var)

    originalVariables = ['Medical_History_1', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
    discreteVariables = list()
    for var in originalVariables:
        if dfTrain[var].isnull().sum():

            discreteVar = var + "_NO_NULLS"
            discreteVariables.append(discreteVar)
            dummyVariable = var + "_IS_NULL"
            dummyVariables.append(dummyVariable)
            dfTrain[dummyVariable] = dfTrain[var].isnull()
            dfTest[dummyVariable] = dfTest[var].isnull()

            nonNullTrain = dfTrain[dfTrain[var].notnull()][var]
            nonNullTest = dfTest[dfTest[var].notnull()][var]
            mle = pd.concat([nonNullTrain, nonNullTest]).mean()

            dfTrain.loc[dfTrain[var].isnull(), discreteVar] = mle
            dfTrain.loc[dfTrain[var].notnull(), discreteVar] = nonNullTrain       
            dfTest.loc[dfTest[var].isnull(), discreteVar] = mle
            dfTest.loc[dfTest[var].notnull(), discreteVar] = nonNullTest

        else:
            discreteVariables.append(var)    
	    
    categoricalVariables = GetCategoricalVariables(10, dfTrain, dfTest)
    
    return continuousVariables + discreteVariables + categoricalVariables + dummyVariables
