
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
from pycirclize.parser import Gff
import numpy as np
import requests
import tempfile
import pickle
import gzip
import gdown
import zipfile
import os



def download_file_from_google_drive(url, destination):
    response = requests.get(url)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        f.write(response.content)

URL_DATASET_1 = 'https://drive.google.com/uc?export=download&id=1fvFcosmNcIxqH0dy56aZU4Cizknm777D'
URL_DATASET_2 = 'https://drive.google.com/uc?export=download&id=1HnFDhSOKwybtD7r9-IwHlwQ6tYHrVChv'
URL_DATASET_3 = 'https://drive.google.com/uc?export=download&id=116E6HD17qkspBEyRZOGE-vRpEyKL0JFu'
url='https://drive.google.com/uc?export=download&id=1MWmwHqhv9QBGvRPwFhgU1JFVuxqmQX7w'
url_1 = "https://drive.google.com/uc?export=download&id=1OWyIwV6JD16CPrMcJ-0ma2kOlzHLCX_y"

@st.cache_data
def load_dataset_1():
    dataset=download_file_from_google_drive(URL_DATASET_1, 'Acession-Numbers.xlsx')
    dataset=pd.read_excel('Acession-Numbers.xlsx', header=1)
    return dataset

@st.cache_data
def load_dataset_2():
    download_file_from_google_drive(URL_DATASET_2, 'Lineage-drug-resitance-classifiation.xlsx')
    return pd.read_excel('Lineage-drug-resitance-classifiation.xlsx', header=1)

@st.cache_data
def load_dataset_3():
    download_file_from_google_drive(URL_DATASET_3, 'WHO-resistance-associated-mutations.xlsx')
    return pd.read_excel('WHO-resistance-associated-mutations.xlsx', header=1)
@st.cache_data
def load_dataset_4():
    download_file_from_google_drive(url, 'genomic.gff')
    return Gff('genomic.gff')

@st.cache_data
def load_dataset_5():

    file_id = '1uA1qLiNrSVSvoVxtOxTB4F6gM5cMOg83'
    url = f"https://drive.google.com/uc?id={file_id}"
    output = 'final_dict.pkl.gz'
    gdown.download(url, output, quiet=False)
    with gzip.open('final_dict.pkl.gz', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_dataset_6():
    
    local_zip_filename = "map_files.zip"
    response = requests.get(url_1)
    with open(local_zip_filename, 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile(local_zip_filename, 'r') as zip_ref:
        zip_ref.extractall("shapefiles")
    shapefile_path = os.path.join("shapefiles", "ne_110m_admin_0_countries.shp")
    map = gpd.read_file(shapefile_path)
    return map


country_origin = load_dataset_1()
lineage_country = load_dataset_2()
resistance_mutations = load_dataset_3()
final_dict = load_dataset_5()
genomic=load_dataset_4()
def go_to_main():
    st.session_state['page'] = 'main'
def go_to_resistance_mutations():
    st.session_state['page'] = 'resistance_mutations'
def go_to_genome_data():
    st.session_state['page'] = 'genome_data'   
def go_to_Countries():
    st.session_state['page'] = 'Countries'

if 'page' not in st.session_state:
    st.session_state['page'] = 'main'
if st.button('Back', key='back'):
  st.session_state['page'] = 'main'

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Countries',key="countries"):
        go_to_Countries()
with col2:
    if st.button('Genome Data', key='genome'):
        go_to_genome_data()
with col3:
    if st.button('Resistance Mutations', key='resistance'):
        go_to_resistance_mutations()

  


if st.session_state['page'] == 'main':
    st.title("Mycobacterium tuberculosis - Africa")
    country_counts = country_origin['Country'].value_counts()
    @st.cache_data
    def load_africa_data():
        map = load_dataset_6()  
        african_countries = [
            'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 
            'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Democratic Republic of the Congo', 
            'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 
            'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 
            'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 
            'Namibia', 'Niger', 'Nigeria', 'Republic of the Congo', 'Rwanda', 'São Tomé and Príncipe', 
            'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 
            'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
        ]
        africa = map[map['SOVEREIGNT'].isin(african_countries)]
        return africa

    def plot_africa_map(africa, country_counts):
        light_cmap = colors.ListedColormap(['#ffcccb', '#ffe4b5', '#fafad2', '#d3ffce', '#add8e6', '#e6e6fa'])

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        africa.plot(ax=ax, cmap=light_cmap, edgecolor='black', linewidth=1.5)

        for idx, row in africa.iterrows():
            country_name = row['SOVEREIGNT']  
            if country_name in country_counts:
                centroid = row['geometry'].centroid
                label = f"{country_name}\n{country_counts[country_name]}"
                ax.text(
                    centroid.x,
                    centroid.y,
                    label,
                    horizontalalignment='center',
                    fontsize=12,
                    color='k',
                    weight='bold'
                )

        plt.title('Number of Occurrences per Country in Africa', fontsize=15)
        return fig

   
    africa = load_africa_data()

   
    fig = plot_africa_map(africa, country_counts)

    
    st.pyplot(fig)
    plt.close()

elif st.session_state['page'] == 'Countries':
    st.subheader("Resistance Mutations")
    filtered_lineage_country = lineage_country.dropna(subset=['Lineage'])
    filtered_lineage_country = filtered_lineage_country[filtered_lineage_country['Lineage'] != '-']
    country_ids = lineage_country['Country'].unique().tolist()
    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox("Select Country", country_ids)
    unique_countries = filtered_lineage_country['Country'].unique()
    filtered_df = filtered_lineage_country[filtered_lineage_country['Country'] == selected_country]
    unique_drugs = filtered_df['Drug'].unique()
    color_map = {drug: color for drug, color in zip(unique_drugs, px.colors.qualitative.Safe)}
    left_col, right_col = st.columns([2, 3])

    with left_col:
        st.subheader(f"Chart for {selected_country}")
        sunburst = px.sunburst(
            filtered_df,
            path=["Drug", "Lineage"],
            values=None,
            color='Drug',
            color_discrete_map=color_map,
            maxdepth=3,
            height=400,
            width=600
        )
        sunburst.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(sunburst, use_container_width=True)

    with right_col:
        fig1 = px.sunburst(
            filtered_lineage_country,
            path=['Lineage'],
            values=None,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            height=250,
            width=250
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.sunburst(
            filtered_lineage_country,
            path=['Drug'],
            values=None,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig2.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            height=250,
            width=250
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.sunburst(
            filtered_lineage_country,
            path=['Country'],
            values=None,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig3.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            height=250,
            width=250
        )
        st.plotly_chart(fig3, use_container_width=True)
 

elif st.session_state['page'] == 'resistance_mutations':
    st.subheader("Resistance Mutations")
    drug_mapping = {
        'AMI': 'Amikacin',
        'BDQ': 'Bedaquiline',
        'CAP': 'Capreomycin',
        'CFZ': 'Clofazimine',
        'DLM': 'Delamanid',
        'EMB': 'Ethambutol',
        'ETH': 'Ethionamide',
        'INH': 'Isoniazid',
        'KAN': 'Kanamycin',
        'LEV': 'Levofloxacin',
        'LZD': 'Linezolid',
        'MXF': 'Moxifloxacin',
        'PZA': 'Pyrazinamide',
        'RIF': 'Rifampicin',
        'STM': 'Streptomycin'}
    resistance_mutations['Drug'] = resistance_mutations['Drug'].replace(drug_mapping)
    drug_position = resistance_mutations.groupby('Drug')['Genomic position '].apply(list).to_dict()
    net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='k', notebook=True, cdn_resources='in_line')
    net.toggle_physics(True)

    added_drugs = set()
    added_variation = set()
    for idx, row in resistance_mutations.iterrows():
        drug = row['Drug']
        gene = row['Gene']
        variation = row['Variation']
        if drug not in added_drugs:
            net.add_node(drug, label=drug, color='green', shape='dot', size=30)
            added_drugs.add(drug)
        if variation not in added_variation:
            net.add_node(variation, label=variation, color='blue', shape='triangle', size=10)
            added_variation.add(variation)
        net.add_edge(drug, variation, label=gene, color='k')

    net.show('drug_gene_network.html')
    st.components.v1.html(open('drug_gene_network.html', 'r').read(), height=750)
 

elif st.session_state['page'] == 'genome_data':
    st.subheader("Genome Data")
    Lineage_dictionary=dict(zip(lineage_country['Name'], lineage_country['Lineage']))
    Drug_dictionary=dict(zip(lineage_country['Name'], lineage_country['Drug']))
    country_dictionary=dict(zip(country_origin['Name'], country_origin['Country']))



    genome_length = 4411532
    drug_position = resistance_mutations.groupby('Drug')['Genomic position '].apply(list).to_dict()
    sample_ids = lineage_country['Name'].unique().tolist()
    selected_sample = st.selectbox("Select Sample ID", sample_ids)



    @st.cache_data
    def plot_original():
      plt.figure(figsize=(50, 10))


      for drug, positions in drug_position.items():
          for pos in set(positions):
              plt.axvline(x=pos, color='red', linewidth=5)


      for i, feat in enumerate(genomic.extract_features("gene")):
          start = int(feat.location.start)
          end = int(feat.location.end)
          length = end - start
          plt.barh(1, length, left=start, color='green', height=1.5)


      plt.xlim(0, genome_length)
      plt.ylim(0.75, 1.25)
      plt.yticks([])

      st.pyplot(plt.gcf())
      plt.close()

    st.write("Places of resistance associated mutations")
    plot_original()

    plt.figure(figsize=(50, 10))

    selected_positions = final_dict.get(selected_sample, [])
    selected_positions = selected_positions.replace('[','').replace(']','').split(',')
    selected_positions = [int(pos.strip()) for pos in selected_positions if pos.strip().isdigit()]
    selected_country=country_dictionary.get(selected_sample)
    selected_lineage=Lineage_dictionary.get(selected_sample)
    selected_drug=Drug_dictionary.get(selected_sample)
    for pos in set(selected_positions):
        plt.axvline(x=int(pos), color='red')



    plt.xlim(0, genome_length)
    plt.ylim(0.75, 1.25)

    plt.yticks([])
    st.write(f"Mutations in: {selected_sample}")
    st.write(f"Country: {selected_country}")
    st.write(f"Lineage: {selected_lineage}")
    st.write(f"Type of resistance: {selected_drug}")

    st.pyplot(plt)
    plt.close()
    
   
