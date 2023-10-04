import pandas

# Training


def fit(x_train, y_train):
    frequency_table = calculate_frecuency_table(x_train, y_train)
    model = model_calculate(frequency_table)
    return model


def calculate_frecuency_table(x_train, y_train):
    # Remove spaces
    classes = y_train.astype(str).str.replace(' ', '')
    # All data (class and attributes)
    all_data = pandas.concat(
        [x_train, classes], axis=1)
    tablas_de_frecuencia = {}
    for column in all_data.columns:
        if column == y_train.name:
            break
        # Remove spaces per column
        all_data[column] = all_data[column].astype(str).str.strip()
        # Frecuency table per column
        # groupby is a Pandas' function, it helps to group data
        # With size(), the functions returns a "Pandas Series" with the number of repeated columns
        # Unstack returns a dataframe, and fill_value helps us to fill missing values
        column_frecuency = all_data.groupby(
            [column, y_train.name]).size().unstack(fill_value=0)
        # orient='index' is used to return a dictionary by instances (rows)
        tablas_de_frecuencia[column] = column_frecuency.to_dict(orient='index')

    return tablas_de_frecuencia


def model_calculate(frecuency_table):
    # This functions expects a dictionary as a parameter with the following structure:
    # {Outlook:{Sunny:{Yes:2,no:3}, Overcast:{Yes:3,No:1}},Temp:{Hot:{Yes:2,No:2},Mild:{Yes:3,No:2}}}
    total_error = {}
    model = None
    aux = {}
    error_rate = 0
    prev_error_rate = None
    # first, we get the atributes (columns)
    for attributes in frecuency_table:
        aux[attributes] = {}
        # Over every atribute we get it's possibles values
        for values_atributes in frecuency_table[attributes]:
            aux[attributes][values_atributes] = list(
                frecuency_table[attributes][values_atributes].keys())[0]
            max_class_value = list(
                frecuency_table[attributes][values_atributes].values())[0]
            total_error = sum(
                frecuency_table[attributes][values_atributes].values())
            # Finally, we obtain the maximum value among the possible classes
            for one_class in frecuency_table[attributes][values_atributes]:
                if max_class_value < frecuency_table[attributes][values_atributes][one_class]:
                    max_class_value = frecuency_table[attributes][values_atributes][one_class]
                    aux[attributes][values_atributes] = one_class
            # Error rate per atribute
            error_rate = error_rate + \
                ((total_error - max_class_value)/total_error)
        # Model values inicializating
        if not model:
            model = {}
            model[attributes] = aux[attributes]
        elif prev_error_rate > error_rate:
            # Deleting and creating the previous model to storage a new model
            model = {}
            model[attributes] = aux[attributes]
        prev_error_rate = error_rate
        # Resetting error rate value
        error_rate = 0
    # This is a model output example (It's a dictionary)
    # {Outlook:{Sunny:Yes,Rainy:No,Overcast:Yes}}
    return model


def tests(x_test, y_test, model):
    all_data = x_test
    # Getting the class
    model_class = list(model.keys())[0]
    classes = y_test.astype(str).str.replace(' ', '')
    # Remove spaces with strip
    for attribute in x_test:
        all_data[attribute] = all_data[attribute].astype(str).str.strip()
    all_data = pandas.concat(
        [all_data, classes], axis=1)
    target_class = y_test.name
    success = 0
    for index, row in all_data.iterrows():
        prediction = model[model_class][row[model_class]]
        # To check if we get a succes or a error
        if prediction == row[target_class]:
            success += 1
    # len() return the number of tested instances
    return len(x_test), success


def print_results(succes, model, total_tests):
    print('**** Modelo ****')
    print(pandas.DataFrame(model))
    print('**** Total acertados ****')
    print(succes)
    print('**** Total probado ****')
    print(total_tests)
    print('**** Porcentaje de aciertos ****')
    print((succes*100)/total_tests)
