import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from wordcloud import WordCloud
import seaborn as sns 


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Youtube Live/Hate Detection Speech')

# Read the dataset from chat.csv which is constantly updating
csv_file_path = 'Updated_Chat.csv'



# Create a function for plotting the data and counting labels
def plot_and_count():
    # Create a plot of the data
    fig, ax = plt.subplots()

    # Create empty lists to store data
    timestamps = []
    labels = []

    # Counter for number of rows
    row_counter = 0

    # Counter for label occurrences
    label_counts = {'hate': 0, 'safe': 0}

    def update_plot():
        nonlocal row_counter

        with open(csv_file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)

            # Extract data from CSV
            for row in csv_reader:
                row_counter += 1

                timestamp_str, _, _, label = row

                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"Skipping invalid timestamp: {timestamp_str}")
                    continue

                # Append data to lists
                timestamps.append(timestamp)
                labels.append(label)

                # Update label counts
                label_counts[label] += 1

        # Plot the data using Matplotlib
        ax.clear()
        ax.plot(timestamps, labels, label='Hate Label')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Hate Label')
        ax.set_title('Live Hate Speech Detection')
        ax.legend()

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig)

        # Close the Matplotlib figure to ensure a fresh figure for the next update
        plt.close(fig)

        # Display information about the data
        st.subheader('Data Statistics')

        # Display total rows
        st.info(f'Total Rows: {row_counter}')

        # Display label counts using bar chart
        st.success('Label Counts (Bar Chart):')
        st.bar_chart(label_counts)

        # Display label counts using pie chart
        st.success('Label Counts (Pie Chart):')
        plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%')
        st.pyplot(plt.gcf())

    # Display some information about the data
    st.subheader('Live Hate Speech Detection')
    st.write('This app displays live updates of hate speech detection in a chat.')

    # Create a button that will update the plot and count labels when clicked
    if st.button('Update Plot and Count Labels'):
        update_plot()

    # Add a space for better separation
    st.write('---')

plot_and_count()


# Set Matplotlib to not show deprecation warnings for plt.pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to generate word cloud
def generate_wordcloud(text):
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud using Matplotlib
    st.subheader('Word Cloud for All Comments')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

# Function to fetch all comments from the CSV file
def get_all_comments():
    # Read the CSV file without column headers
    df = pd.read_csv(csv_file_path, header=None, names=range(5))  # Assuming there are 3 columns
    return " ".join(df[2].astype(str))  # Use column index 2 (adjust as needed)

# Function to get all comments which are safe 
def get_safe_comments():
    # Read the CSV file without column headers
    df = pd.read_csv(csv_file_path, header=None, names=range(5))  # Assuming there are 3 columns
    return " ".join(df[df[3] == 'safe'][2].astype(str))  # Use column index 2 (adjust as needed)

# Display word cloud button
if st.button('Generate Word Cloud for All Comments'):
    all_comments_text = get_all_comments()
    generate_wordcloud(all_comments_text)

# Add a space for better separation
st.write('---')


if st.button('Generate Word Cloud for Safe Comments(Bloom Filter)'):
    safe_comments_text = get_safe_comments()
    generate_wordcloud(safe_comments_text)

def heatmap():
    # Create empty lists to store data
    timestamps = []
    labels = []

    # Counter for number of rows
    row_counter = 0

    def update_plot():
        nonlocal row_counter

        with open(csv_file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file)

            # Extract data from CSV
            for row in csv_reader:
                row_counter += 1

                timestamp_str, _, _, label = row

                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"Skipping invalid timestamp: {timestamp_str}")
                    continue

                # Append data to lists
                timestamps.append(timestamp)
                labels.append(label)

        # Plot heatmap using Seaborn for density of hate comments over time
        df = pd.DataFrame({'timestamps': timestamps, 'labels': labels})
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df['time'] = df['timestamps'].dt.time
        df['hour'] = df['timestamps'].dt.hour
        df['minute'] = df['timestamps'].dt.minute
        df['second'] = df['timestamps'].dt.second

        # Create a pivot table
        pivot = df.pivot_table(index=['hour', 'minute'], columns='time', values='labels', aggfunc='count')

        # Plot the heatmap
        st.subheader('Heatmap of Hate Comments Over Time')
        sns.heatmap(pivot, cmap='coolwarm', linecolor='white', linewidth=1)
        st.pyplot()

    # Display some information about the data
    st.subheader('Live Hate Speech Detection')
    st.write('This app displays live updates of hate speech detection in a chat.')

    # Create a button that will update the plot and count labels when clicked
    if st.button('Update Heatmap'):
        update_plot()

    # Add a space for better separation
    st.write('---')

heatmap()
