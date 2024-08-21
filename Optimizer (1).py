#pip install streamlit sqlalchemy pyodbc pandas numpy scikit-learn

import streamlit as st
from sqlalchemy import create_engine
import pyodbc
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

# In[3]:


server = 'XXXX'
database = 'XXXX'
username = 'XXXX'
password = 'XXXXX'
engine = create_engine(f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server')
query1 = 'SELECT * FROM Location_Planograms_Overview;'
df_Overview = pd.read_sql(query1, engine)
query2 = 'SELECT * FROM Planogram_Averages;'
df_Avg = pd.read_sql(query2, engine)
query3 = 'SELECT * FROM Location_Adjustments;'
df_Adjust = pd.read_sql(query3, engine)

# In[4]:


df_Overview = df_Overview.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df_Overview = df_Overview.applymap(
    lambda x: str(x).replace('$', '').replace(',', '').replace('-', '') if isinstance(x, str) else x)
df_Overview = df_Overview.applymap(lambda x: np.nan if x == '' else x)
df_Overview = df_Overview.fillna(0)

# In[63]:


df_Adjust = df_Adjust.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# print(df_Adjust.dtypes)


# In[6]:


df_Avg = df_Avg.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# In[7]:


df_Overview['Avg_Wly_Profit_Sales'] = df_Overview['Avg_Wly_Profit_Sales'].astype(float)

scaler = MinMaxScaler()


def added_columns(df_store_dept, fixture, lowest_fixture_width):
    df_store_dept['ppp'] = df_store_dept['Avg_Wly_Profit_Sales'] / df_store_dept['Planogram_Width(ft)']
    dept_size = df_store_dept['Planogram_Width(ft)'].sum()
    df_store_dept['modified_pl_length'] = df_store_dept.apply(
        lambda row: (row['ppp'] / df_store_dept['ppp'].sum()) * dept_size, axis=1)
    df_fix = df_store_dept[df_store_dept['Fixture_Name'] == fixture]
    df_fix['round_length'] = df_fix['modified_pl_length'].apply(lambda row: math.ceil(row))
    df_fix['csi'] = df_fix['Avg_Wly_Unit_Sales'] / df_store_dept['Avg_Wly_Unit_Sales'].sum()
    df_fix[['csi']] = scaler.fit_transform(df_fix[['csi']]).round(2)
    fix_less_rows = df_fix[df_fix['modified_pl_length'] > 0]
    #     fix_less_rows['csi'] = fix_less_rows['Avg_Wly_Unit_Sales']/df_store_dept['Avg_Wly_Unit_Sales'].sum()

    data = {
        'Product': fix_less_rows['Planogram_Name'].tolist(),
        'Pl_Width': fix_less_rows['Planogram_Width(ft)'].tolist(),
        'Profit': fix_less_rows['ppp'].astype(float).tolist(),
        'round_length': fix_less_rows['round_length'].tolist(),
        'csi': fix_less_rows['csi'].tolist()
    }

    df = pd.DataFrame(data)
    return df


# In[113]:


def fractional_knapsack(df, capacity):
    df['ValuePerWeight'] = df['Profit'] / df['round_length']
    df = df.sort_values(by='ValuePerWeight', ascending=False)

    total_profit = 0.0
    current_weight = 0.0
    selected_products = []

    for index, row in df.iterrows():
        weight = row['round_length']
        profit = row['Profit']
        if current_weight + weight <= capacity:
            selected_products.append(row.to_dict())  # Convert row to dictionary
            current_weight += weight
            total_profit += profit
        else:
            remaining_capacity = capacity - current_weight
            fraction = remaining_capacity / weight
            selected_products.append({
                'Product': row['Product'],
                'Pl_Width': row['Pl_Width'],
                'round_length': row['round_length'] * fraction,
                'Profit': row['Profit'] * fraction,
                'csi': row['csi']
            })
            total_profit += fraction * profit
            break

    return total_profit, selected_products


# In[114]:
def knapsack(df, capacity):
    df = df.sort_values(by='round_length')  # Sort by weight in ascending order

    n = len(df)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight = df.iloc[i - 1]['round_length']
        profit = df.iloc[i - 1]['Profit']

        for w in range(1, capacity + 1):
            if weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + profit)
            else:
                dp[i][w] = dp[i - 1][w]

    total_profit = dp[n][capacity]
    selected_products = []

    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_products.append(df.iloc[i - 1].to_dict())
            w -= df.iloc[i - 1]['round_length']

    return total_profit, selected_products[::-1]  # Reverse the order of selected products


def unbounded_knapsack(df, capacity):
    df['ValuePerWeight'] = df['Profit'] / df['round_length']
    df = df.sort_values(by='ValuePerWeight', ascending=False)

    total_profit = 0.0
    count_dict = {product['Product']: 0 for _, product in df.iterrows()}
    selected_products = []
    current_weight = 0.0  # Initialize current_weight here

    for index, row in df.iterrows():
        weight = row['round_length']
        profit = row['Profit']
        #####################################Unbounded Knapsack
        while count_dict[row['Product']] < 2 and current_weight + weight <= capacity:  ###Change frequency here
            selected_products.append({
                'Product': row['Product'],
                'Pl_Width': row['Pl_Width'],
                'round_length': row['round_length'],
                'Profit': row['Profit'],
                'csi': row['csi']
            })
            current_weight += weight
            total_profit += profit
            count_dict[row['Product']] += 1

    return total_profit, selected_products


class SpaceFitter1:
    def __init__(self, num_width, width):
        self.num_width = num_width
        self.width = width

    def find_coefficients_with_limits(self, n):
        # Check if it's possible to allocate n using the given width and number
        if n % self.width == 0 and n // self.width <= self.num_width:
            return n // self.width
        return None

    def fit_space_with_limits(self, df):
        df['Number of fixtures'] = 0

        for index, row in df.iterrows():
            space = row['New Width']
            coefficients = self.find_coefficients_with_limits(space)

            if coefficients is not None:
                num_allocations = coefficients
                df.at[index, 'Number of fixtures'] = num_allocations
                self.num_width -= num_allocations

        return df

    def update_available_space(self, df):
        # Create a copy of the input DataFrame
        dummy_df = df.copy()

        # Identify rows where Number_of_width is 0
        rows_to_update = dummy_df[dummy_df['Number of fixtures'] == 0].index

        if self.num_width != 0:
            # Update available space for rows with zero allocations
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[rows_to_update, 'New Width'].apply(
                lambda x: round(x / self.width) * self.width)

        # Rerun the allocation process for the updated rows only on the dummy DataFrame
        dummy_df = self.fit_space_with_limits(dummy_df.loc[rows_to_update])

        # Update the original DataFrame with the modified rows from the dummy DataFrame
        df.loc[rows_to_update] = dummy_df

        return df

class SpaceFitter2:
    def __init__(self, num_width1, num_width2, width1, width2):
        self.num_width1 = num_width1
        self.num_width2 = num_width2
        self.width1 = width1
        self.width2 = width2
    def find_coefficients_with_limits(self, n):
        for a in range(min(n // self.width1, self.num_width1) + 1):
            if (n - self.width1 * a) % self.width2 == 0 and (n - self.width1 * a) // self.width2 <= self.num_width2:
                b = (n - self.width1 * a) // self.width2
                return a, b
        return None
    def fit_space_with_limits(self, df):
        df['Number_of_width1'] = 0
        df['Number_of_width2'] = 0
        for index, row in df.iterrows():
            space = row['New Width']
            coefficients = self.find_coefficients_with_limits(space)
            if coefficients:
                a, b = coefficients
                df.at[index, 'Number_of_width1'] = a
                df.at[index, 'Number_of_width2'] = b
                self.num_width1 -= a
                self.num_width2 -= b
        return df

    def update_available_space(self, df):

    # Create a copy of the input DataFrame
        dummy_df = df.copy()
    # Identify rows where both Number_of_width1 and Number_of_width2 are 0
        rows_to_update = dummy_df[(dummy_df['Number_of_width1'] == 0) & (dummy_df['Number_of_width2'] == 0)].index
        if self.num_width1 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[rows_to_update, 'New Width'].apply(lambda x: round(x / self.width1) * self.width1)
        elif self.num_width2 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[rows_to_update, 'New Width'].apply(lambda x: round(x / self.width2) * self.width2)
    # Rerun the allocation process for the updated rows only on the dummy DataFrame
        dummy_df = self.fit_space_with_limits(dummy_df.loc[rows_to_update])
    # Update the original DataFrame with the modified rows from the dummy DataFrame
        df.loc[rows_to_update] = dummy_df

        return df


class SpaceFitter3:
    def __init__(self, num_width1, num_width2, num_width3, width1, width2, width3):
        self.num_width1 = num_width1
        self.num_width2 = num_width2
        self.num_width3 = num_width3
        self.width1 = width1
        self.width2 = width2
        self.width3 = width3

    def find_coefficients_with_limits(self, n):
        for a in range(min(n // self.width1, self.num_width1) + 1):
            for b in range(min((n - self.width1 * a) // self.width2, self.num_width2) + 1):
                if (n - self.width1 * a - self.width2 * b) % self.width3 == 0 and \
                        (n - self.width1 * a - self.width2 * b) // self.width3 <= self.num_width3:
                    c = (n - self.width1 * a - self.width2 * b) // self.width3
                    return a, b, c
        return None

    def fit_space_with_limits(self, df):
        df['Number_of_width1'] = 0
        df['Number_of_width2'] = 0
        df['Number_of_width3'] = 0

        for index, row in df.iterrows():
            space = row['New Width']
            coefficients = self.find_coefficients_with_limits(space)

            if coefficients:
                a, b, c = coefficients
                df.at[index, 'Number_of_width1'] = a
                df.at[index, 'Number_of_width2'] = b
                df.at[index, 'Number_of_width3'] = c
                self.num_width1 -= a
                self.num_width2 -= b
                self.num_width3 -= c

        return df

    def update_available_space(self, df):
        # Create a copy of the input DataFrame
        dummy_df = df.copy()

        # Identify rows where all Number_of_widths are 0
        rows_to_update = dummy_df[
            (dummy_df['Number_of_width1'] == 0) & (dummy_df['Number_of_width2'] == 0) & (
                        dummy_df['Number_of_width3'] == 0)].index

        if self.num_width1 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width1) * self.width1)
        elif self.num_width2 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width2) * self.width2)
        elif self.num_width3 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width3) * self.width3)

        # Rerun the allocation process for the updated rows only on the dummy DataFrame
        dummy_df = self.fit_space_with_limits(dummy_df.loc[rows_to_update])

        # Update the original DataFrame with the modified rows from the dummy DataFrame
        df.loc[rows_to_update] = dummy_df

        return df

class SpaceFitter4:
    def __init__(self, num_width1, num_width2, num_width3, num_width4, width1, width2, width3, width4):
        self.num_width1 = num_width1
        self.num_width2 = num_width2
        self.num_width3 = num_width3
        self.num_width4 = num_width4
        self.width1 = width1
        self.width2 = width2
        self.width3 = width3
        self.width4 = width4

    def find_coefficients_with_limits(self, n):
        for a in range(min(n // self.width1, self.num_width1) + 1):
            for b in range(min((n - self.width1 * a) // self.width2, self.num_width2) + 1):
                for c in range(min((n - self.width1 * a - self.width2 * b) // self.width3, self.num_width3) + 1):
                    if (n - self.width1 * a - self.width2 * b - self.width3 * c) % self.width4 == 0 and \
                            (n - self.width1 * a - self.width2 * b - self.width3 * c) // self.width4 <= self.num_width4:
                        d = (n - self.width1 * a - self.width2 * b - self.width3 * c) // self.width4
                        return a, b, c, d
        return None

    def fit_space_with_limits(self, df):
        df['Number_of_width1'] = 0
        df['Number_of_width2'] = 0
        df['Number_of_width3'] = 0
        df['Number_of_width4'] = 0

        for index, row in df.iterrows():
            space = row['New Width']
            coefficients = self.find_coefficients_with_limits(space)

            if coefficients:
                a, b, c, d = coefficients
                df.at[index, 'Number_of_width1'] = a
                df.at[index, 'Number_of_width2'] = b
                df.at[index, 'Number_of_width3'] = c
                df.at[index, 'Number_of_width4'] = d
                self.num_width1 -= a
                self.num_width2 -= b
                self.num_width3 -= c
                self.num_width4 -= d

        return df

    def update_available_space(self, df):
        # Create a copy of the input DataFrame
        dummy_df = df.copy()

        # Identify rows where all Number_of_widths are 0
        rows_to_update = dummy_df[
            (dummy_df['Number_of_width1'] == 0) & (dummy_df['Number_of_width2'] == 0) & (
                        dummy_df['Number_of_width3'] == 0)
            & (dummy_df['Number_of_width4'] == 0)].index

        # Update 'New Width' based on remaining widths
        if self.num_width1 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width1) * self.width1)
        elif self.num_width2 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width2) * self.width2)
        elif self.num_width3 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width3) * self.width3)
        elif self.num_width4 != 0:
            dummy_df.loc[rows_to_update, 'New Width'] = dummy_df.loc[
                rows_to_update, 'New Width'].apply(lambda x: round(x / self.width4) * self.width4)

        # Rerun the allocation process for the updated rows only on the dummy DataFrame
        dummy_df = self.fit_space_with_limits(dummy_df.loc[rows_to_update])

        # Update the original DataFrame with the modified rows from the dummy DataFrame
        df.loc[rows_to_update] = dummy_df

        return df

def main():

    with st.sidebar:
        store_code = st.text_input("Type Store Code")
        # dept_name = st.text_input("Type department name")
        dept_name = st.selectbox("Select department name", df_Overview['Department_Name'].unique())
        fix_type = st.selectbox("Select fixture name", df_Overview['Fixture_Name'].unique())
        # fix_length = st.selectbox("Select department name", df_Overview['Fixture_Name'].unique())
        # Get input for the first list of numbers
        width = st.text_input("Enter the types of fixtures:")

        # Convert the input string into a list of numbers
        width = [int(num) for num in width.split()]

        # Get input for the second list of numbers
        num = st.text_input("Enter their frequency")

        # Convert the input string into a list of numbers
        num = [int(num) for num in num.split()]



        capacity = int(st.number_input("Capacity"))

        submit_button = st.button("Submit")

    if submit_button:
        df_store = df_Overview[df_Overview['Store_Code'] == store_code]
        df_store_dept = df_store[df_store['Department_Name'] == dept_name]

        # Assuming knapsack and other functions are defined
        max_profit, selected_products = knapsack(added_columns(df_store_dept, fix_type, 0), capacity)

        # Order selected products by ppp
        selected_products.sort(key=lambda x: x['Profit'], reverse=True)

        # Display selected products using Streamlit

        count = 0
        total_profit = 0
        total_space = 0

        round_width_list = []
        ppp_list = []

        data = []
        for product in selected_products:
            count += 1
            total_space += product['round_length']
            # st.write(f"Planogram_Name: {product['Product']}\n'round_mod_Width': {product['round_length']}, OG_Width: {product['Pl_Width']}, ppp: {round(product['Profit'], 2)}, csi: {product['csi']}")
            total_profit += round(product['Profit'] * product['round_length'], 2)

            data.append({
                'Planogram_Name': product['Product'],
                'New Width': product['round_length'],
                'Original Width': product['Pl_Width'],
                'Profit Per Foot': round(product['Profit'], 2),
                'CSI': product['csi']
            })

            round_width_list.append(product['round_length'])

            # Create a Pandas DataFrame
            result_df = pd.DataFrame(data)

            # Create an instance of SpaceFitter with initial parameters
            if(len(num)==1):
                space_fitter = SpaceFitter1(*num, *width)
            elif(len(num)==2):
                space_fitter = SpaceFitter2(*num, *width)
            elif(len(num)==3):
                space_fitter = SpaceFitter3(*num, *width)
            else:
                space_fitter = SpaceFitter4(*num, *width)


            # Fit space with limits based on the available space in the DataFrame
        df = space_fitter.fit_space_with_limits(pd.DataFrame({'New Width': round_width_list}))

            # Update available space in the DataFrame
        df = space_fitter.update_available_space(df)

        df['Planogram_Name'] = result_df['Planogram_Name']
        df['Original Width'] = result_df['Original Width']
        df['Profit Per Foot'] = result_df['Profit Per Foot']
        df['CSI'] = result_df['CSI']

            # Display selected products DataFrame using Streamlit
        st.write("Selected Products:")


        # st.table(result_df)
        st.table(df)

        df['total_profit'] = df['Profit Per Foot']*df['New Width']
        total_profit = round(df['total_profit'].sum(), 2)

        # ppp_list.append(product['Profit'])
        med_csi = round(df['CSI'].median(), 2)
        st.write(f"Count of planograms : {count}")
        st.write(f"Total Profit: {total_profit}")
        st.write(f"CSI Median: {med_csi}")
        # st.write(f"Total Space: {total_space}")


if __name__ == "__main__":
    main()
