import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import statsmodels.api as sm


st.set_page_config(layout="wide")

st.title("Capstone Data Exploration")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Daily Activity', "Cluster Analysis", "Cluster Time Series Data", "Correlation", "Regression"])


def column_filter_single(df, option, column_mapping, common_columns):
    selected_columns = []

    option = option.split()[0].lower()
    if option in column_mapping:
        selected_columns.extend(column_mapping[option])
    
    final_columns = list(set(common_columns + selected_columns))  # Ensure uniqueness
    df = df[final_columns]

    return df, selected_columns, final_columns


def column_filter_multiple(df, options, column_mapping, common_columns):
    selected_columns = []

    for option in options:
        option = option.split()[0].lower()
        if option in column_mapping:
            selected_columns.extend(column_mapping[option])
    
    final_columns = list(set(common_columns + selected_columns))  # Ensure uniqueness
    df = df[final_columns]

    return df, selected_columns, final_columns

def date_filter(df, date_range):
    df = df[(df["ActivityDate"] >= date_range[0]) & (data["ActivityDate"] <= date_range[1])]
    return df

#DAILY ACTIVTIY
with tab1:
    df = pd.read_csv("cleaned_data/DailyActivity.csv")

    #Filter columns
    columns = ["Steps", "Calories", "Tracked Distance", "Distance Breakdown", "Minutes Breakdown"]
    column_option = st.pills("Y-axis", columns, selection_mode="single", default="Steps")
    column_mapping = {"steps": ["TotalSteps"], "calories": ["Calories"], "tracked": ["TotalDistance", "LoggedActivitiesDistance"], "distance": ["VeryActiveDistance", "ModeratelyActiveDistance", "LightActiveDistance", "SedentaryActiveDistance"], "minutes": ["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]}
    data, selected_columns, final_columns = column_filter_single(df, column_option, column_mapping, ['Id', 'ActivityDate', 'Day_of_Week'])
    
    
    col1, col2, col3 = st.columns([3,0.09,0.9])

    with col3:
        #Filter by ID
        id_list = list(data['Id'].unique())
        id = st.selectbox("Select an ID", id_list)
        data = data[data["Id"]==id]
        st.write("")

        show_avg = st.segmented_control("Display Average", ["On", "Off"], selection_mode="single", default="Off")
        st.write("")

        date_option = st.radio("Date Aggregation" ,["Daily", "Day of Week"], horizontal=True)
        
        if date_option == "Daily":
            #Filter by Date
            oldest_date = data['ActivityDate'].iloc[0]
            latest_date = data['ActivityDate'].iloc[-1]
            date_list = data['ActivityDate'].unique()

            date_range = st.select_slider("Date Range", options=date_list, value = (oldest_date, latest_date))
            data = date_filter(data, date_range)
            dateaxis = "ActivityDate"
        else:
            data = data.groupby('Day_of_Week')[selected_columns].mean().reset_index()

            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            data['Day_of_Week'] = pd.Categorical(data['Day_of_Week'], categories=day_order, ordered=True)

            # Sort by the defined order
            data = data.sort_values('Day_of_Week')
            dateaxis = "Day_of_Week"


    with col1:
        fig = px.line(data, x=dateaxis, y=selected_columns)
        fig.update_layout(yaxis = dict(showgrid=False))
        #fig.update_layout(legend=dict(x=0, y=1, xanchor='left', yanchor='top'))
        fig.update_layout(showlegend = False, yaxis_title="")

        if show_avg == "On":
            if dateaxis == "ActivityDate":
                average = df[(df["ActivityDate"] >= date_range[0]) & (df["ActivityDate"] <= date_range[1])][final_columns].drop(columns="Day_of_Week")
                average = average.groupby("ActivityDate").mean().reset_index()
            else:
                average = df[final_columns].drop(columns="ActivityDate").groupby("Day_of_Week").mean().reset_index()
                average['Day_of_Week'] = pd.Categorical(average['Day_of_Week'], categories=day_order, ordered=True)
                average = average.sort_values('Day_of_Week')
            fig_avg = px.line(average, x=dateaxis, y=selected_columns)

            # Add each trace from fig_avg to fig
            for trace in fig_avg.data:
                trace.line = dict(dash="dot", width=2)
                fig.add_trace(trace)

        for col in selected_columns:
            last_x = data[dateaxis].iloc[0]  # Last x-value (date)
            last_y = data[col].iloc[0]  # Last y-value (line end)

            fig.add_annotation(
                x=last_x,
                y=last_y,
                text=col,  # Use the column name as the label
                showarrow=True,
                arrowwidth=1,
                font=dict(size=12, color="black"),
                xanchor="right",
                yanchor="middle"
            )

        st.plotly_chart(fig)

    with st.expander("View Data"):
        st.dataframe(data)

#CLUSTER 3D PLOT
with tab2:
    clusters = pd.read_csv("cleaned_data/Clusters.csv")
    df =  pd.read_csv("cleaned_data/Customer_Profile.csv") 
    
    clusters['Cluster'] = clusters['Cluster'] + 1

    c1, c2, c3, c4 = st.columns(4)
    counts = df['Cluster'].value_counts().sort_index(ascending=True)
    c1.metric('Cluster 1', value=str(counts.iloc[0]+1) + " customers")
    c2.metric('Cluster 2', value=str(counts.iloc[1]+1) + " customers")
    c3.metric('Cluster 3', value=str(counts.iloc[2]+1) + " customers")
    c4.metric('Cluster 4', value=str(counts.iloc[3]+1) + " customers")

    col1, col2, col3 = st.columns([1.5,1, 0.1])

    with col2:
        columns = ["Steps", "Calories", "Intensity", "Distance", "Time Slept", "Sleep Quality", "Weight", "BMI", "METs"]

        column_dict = {"steps": ["TotalSteps"], 
                       "calories": ["Calories"], 
                       "distance": ["TotalDistance"], 
                       "time": ['TotalTimeInBed'], 
                       'sleep':['SleepQuality'], 
                       'weight':['WeightKg'], 
                       'bmi':['BMI'], 
                       'intensity':['avg_intensity'],
                       'mets':['METs']}

        # Select X-axis
        x = st.pills("Select X-axis", columns, default="Steps", key="x")
        st.divider()

        # Select Y-axis dynamically (removing X)
        y = st.pills("Select Y-axis", columns, default="Calories", key="y")
        st.divider()
        
        # Select Z-axis dynamically (removing X and Y)
        z = st.pills("Select Z-axis", columns, default="Intensity", key="z")

    with col1:
    
        data, cols, final_cols = column_filter_multiple(clusters, [x,y,z], column_dict, ['Cluster'])
        fig = px.scatter_3d(data, x=cols[0], y=cols[1], z=cols[2], color='Cluster')

        # Adjust layout for padding (increase margins)
        fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, b=60, t=20),
            showlegend=False,
            scene=dict(
                xaxis_title=cols[0],
                yaxis_title=cols[1],
                zaxis_title=cols[2]
            )
        )


        # Add annotations for each cluster
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]
            mean_x = cluster_data[cols[0]].mean()
            mean_y = cluster_data[cols[1]].mean()
            mean_z = cluster_data[cols[2]].mean()

            fig.add_trace(go.Scatter3d(
                x=[mean_x], y=[mean_y], z=[mean_z],
                text=[f"Cluster {cluster}"],
                mode='text',
                textposition="middle center",
                showlegend=False
            ))

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

    footerleft, footerright = st.columns(2)

    with footerleft:
        with st.expander("View Cluster Means Data"):
            st.dataframe(clusters)
    
    with footerright:
        with st.expander("View Customer Profile Data"):
            st.dataframe(df)

#CLUSTER TIME SERIES DATA
with tab3:
    df = pd.read_csv("cleaned_data/DailyActivity_Clustered.csv")

    cluster = df.drop(columns=["Id", "Day_of_Week"]).groupby(["ActivityDate", "Cluster"]).mean().reset_index()
    cluster_day = df.drop(columns=["Id", "ActivityDate"]).groupby(["Day_of_Week", "Cluster"]).mean().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    cluster_day['Day_of_Week'] = pd.Categorical(cluster_day['Day_of_Week'], categories=day_order, ordered=True)
    cluster_day = cluster_day.sort_values('Day_of_Week')

    profile =  pd.read_csv("cleaned_data/Customer_Profile.csv") 

    c1, c2, c3, c4 = st.columns(4)
    counts = profile['Cluster'].value_counts().sort_index(ascending=True)
    c1.metric('Cluster 1', value=str(counts.iloc[0]+1) + " customers")
    c2.metric('Cluster 2', value=str(counts.iloc[1]+1) + " customers")
    c3.metric('Cluster 3', value=str(counts.iloc[2]+1) + " customers")
    c4.metric('Cluster 4', value=str(counts.iloc[3]+1) + " customers")

    col1, col2 = st.columns([2,1])

    with col2:
        time = st.segmented_control("", ["Date", "Day of Week"], default="Date", selection_mode="single")
        option = st.pills("Select a column", options=cluster.drop(columns=["ActivityDate", "Cluster"]).columns, selection_mode="single", default="TotalSteps")
        
    with col1:    
        if time == "Date":
            fig3 = px.line(cluster, x="ActivityDate", y=option, color='Cluster')
            fig3.update_layout(yaxis = dict(showgrid=False))
        elif time == "Day of Week":
            fig3 = px.line(cluster_day, x="Day_of_Week", y=option, color='Cluster')
            fig3.update_layout(yaxis = dict(showgrid=False))

        
        st.plotly_chart(fig3)

#Correlation
with tab4:
    df = pd.read_csv("cleaned_data/Customer_Profile.csv")

    df = df.drop(columns=["Id", "NaN", "Cluster"]).reset_index().drop(columns="index")

    corr_matrix = df.corr()

    fig = ff.create_annotated_heatmap(
    x=corr_matrix.columns.tolist(), 
    y=corr_matrix.columns.tolist(),
    z=corr_matrix.values,  
    annotation_text=np.round(corr_matrix.values, 2),  # Round for readability
    colorscale="Reds",  # Change to "Blues", "Plasma", "Cividis", etc.
    showscale=True
    )

    # Update Layout for Aesthetic Look
    fig.update_layout(
        xaxis=dict(tickangle=-45, side="top"),  # Rotate labels for clarity
        width=50,
        height=800,
        font=dict(size=12),
        showlegend=False
    )

    st.plotly_chart(fig)

    with st.expander("View Data"):
        st.dataframe(df)


with tab5:

        df = pd.read_csv("cleaned_data/Customer_Profile.csv")

        if 'reg_var' not in st.session_state:
            st.session_state.reg_var = df.drop(columns=["Id", "Cluster", "NaN"]).columns
        
        c1, c3, c2 = st.columns([1, 0.2,2])

        with c1:
            with st.expander("Target (Dependent) Variable"):
                dependent = st.pills(label="",options=df.drop(columns=["Id", "Cluster", "NaN", "most_intense_day"]).columns, selection_mode="single", default="TotalSteps")


        with c1:
            with st.expander("Predictor (Independent) Variables"):
                independent = st.pills("",options=df.drop(columns=["Id", "Cluster", "NaN", 'most_intense_day']).columns, selection_mode="multi", default="TotalDistance")
        
        c1.divider()

        cluster_option = c1.segmented_control("Select Cluster/s", [1, 2, 3, 4], default=[1,2,3,4], selection_mode="multi")
        cluster_option  = list(map(lambda x: x - 1, cluster_option))

        c1.divider()
    

        if dependent and independent and cluster_option:
            
            data = df[df['Cluster'].isin(cluster_option)]

            # Add constant for intercept
            X = sm.add_constant(data[independent])  
            y = data[dependent]

            # Fit the model
            model = sm.OLS(y, X).fit()

            coefficients = model.params

            # Get intercept (constant term)
            intercept = coefficients[0]

            # Get independent variables
            independent_vars = coefficients.index[1:]  # Exclude intercept

            # Create equation string
            equation = f"Y = {intercept:.4f} "  # Round for readability
            for var in independent_vars:
                coef = coefficients[var]
                equation += f"+ ({coef:.4f} * {var}) "

            yaxis = c1.pills("Select Predictor to plot against "+dependent, independent, selection_mode="single", default=independent[0])
            x_plot = data[dependent]
            y_plot = data[yaxis]
            coefficients = np.polyfit(x_plot, y_plot, 1)  # 1 for linear fit
            polynomial = np.poly1d(coefficients)
            y_hat = polynomial(x_plot)

            correlation = np.corrcoef(x_plot, y_plot)[0, 1]

            fig = px.scatter(data, x=dependent, y=yaxis, title=yaxis+" v/s "+dependent, height=350)
            fig.update_traces(marker=dict(color='black'))
            fig.add_trace(go.Scatter(x=x_plot, y=y_hat, mode='lines', showlegend=False, marker=dict(color='red')))
            fig.update_layout(
                    autosize=True,
                    margin=dict(l=50, r=50, b=0, t=50),
                    showlegend=False,
                    scene=dict(
                        xaxis_title=cols[0],
                        yaxis_title=cols[1],
                        zaxis_title=cols[2]
                    )
                )
            
            with c2:
                #fig.add_trace(go.Scatter(x=df[dependent], y=df['y_pred'], mode='lines', name='OLS Fit Line', line=dict(color='red', width=3)))
                leftcol, rightcol = st.columns([3, 1])
                leftcol.plotly_chart(fig)
                rightcol.metric("Correlation Co-efficient", value=round(correlation, 3), border=True)
                rightcol.metric("R-Squared Value", value=round(model.rsquared, 3), border=True)

                if model.rsquared < 0.3:
                    rightcol.metric("Model Strength", value="Weak",border=True)
                elif model.rsquared > 0.3 and model.rsquared < 0.7:
                    rightcol.metric("Model Strength", value="Moderate", border=True)
                elif model.rsquared > 0.7 and model.rsquared < 0.9:
                    rightcol.metric("Model Strength", value="Strong", border=True)
                else:
                    rightcol.metric("Model Strength", value="Perfect", border=True)
                
                with st.container(border=True):
                    st.markdown("**Regression Equation:**")
                    st.write(equation)




            # Summary with p-values, R-squared, etc.
            with st.expander("Stats Summary for Nerds"):
                st.write(model.summary())
        else:
            c2.error("Make at least one selection for each input")

        


# WORK ON DAY OF WEEK FOR CLUSTER TIME SERIES
