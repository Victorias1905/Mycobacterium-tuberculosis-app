%%writefile app.py

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
import pickle
import os

# Function to download files from Google Drive
def download_file_from_google_drive(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        st.write(f"File downloaded successfully: {destination}")
    else:
        st.error(f"Failed to download the file from Google Drive. Status code: {response.status_code}")

# URLs for the files on Google Drive (converted to direct download links)
URL_DATASET_1 = 'https://drive.google.com/uc?export=download&id=1fvFcosmNcIxqH0dy56aZU4Cizknm777D'
URL_DATASET_2 = 'https://drive.google.com/uc?export=download&id=1HnFDhSOKwybtD7r9-IwHlwQ6tYHrVChv'
URL_DATASET_3 = 'https://drive.google.com/uc?export=download&id=116E6HD17qkspBEyRZOGE-vRpEyKL0JFu'
URL_DATASET_5 = 'https://drive.google.com/uc?export=download&id=1wV0PnquESSPVv1xiJ8TgNXsDHvyzmFyM'
URL_GFF_FILE = 'https://drive.google.com/uc?export=download&id=1yAq-K1VdJF1t0wrE-p787WmX-mJqDIEf'

# Functions to load datasets

def load_dataset_1():
    file_path = 'Acession-Numbers.xlsx'
    download_file_from_google_drive(URL_DATASET_1, file_path)
    if os.path.exists(file_path):
        return pd.read_excel(file_path, header=1)
    else:
        st.error(f"Failed to load dataset: {file_path}")
        return None

def load_dataset_2():
    file_path = 'Lineage-drug-resitance-classifiation.xlsx'
    download_file_from_google_drive(URL_DATASET_2, file_path)
    if os.path.exists(file_path):
        return pd.read_excel(file_path, header=1)
    else:
        st.error(f"Failed to load dataset: {file_path}")
        return None

def load_dataset_3():
    file_path = 'WHO-resistance-associated-mutations.xlsx'
    download_file_from_google_drive(URL_DATASET_3, file_path)
    if os.path.exists(file_path):
        return pd.read_excel(file_path, header=1)
    else:
        st.error(f"Failed to load dataset: {file_path}")
        return None

def load_dataset_5():
    file_path = 'final_dict.pkl'
    download_file_from_google_drive(URL_DATASET_5, file_path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error(f"Failed to load dataset: {file_path}")
        return None

def load_gff_file():
    file_path = 'genomic.gff'
    download_file_from_google_drive(URL_GFF_FILE, file_path)
    if os.path.exists(file_path):
        return Gff(file_path)
    else:
        st.error(f"Failed to load GFF file: {file_path}")
        return None

# Navigation functions
def go_to_main():
    st.session_state['page'] = 'main'

def go_to_resistance_mutations():
    st.session_state['page'] = 'resistance_mutations'

def go_to_genome_data():
    st.session_state['page'] = 'genome_data'

if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Genome Data', key='genome'):
        go_to_genome_data()
with col2:
    if st.button('Resistance Mutations', key='resistance'):
        go_to_resistance_mutations()
with col3:
    if st.button('Back'):
        st.session_state['page'] = 'main'

if st.session_state['page'] == 'main':
    st.title("Mycobacterium tuberculosis - Africa")
    country_origin = load_dataset_1()
    if country_origin is not None:
        country_counts = country_origin['Country'].value_counts()

        lineage_country = load_dataset_2()
        if lineage_country is not None:
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

            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            africa = world[(world['continent'] == 'Africa')]
            light_cmap = colors.ListedColormap(['#ffcccb', '#ffe4b5', '#fafad2', '#d3ffce', '#add8e6', '#e6e6fa'])

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            africa.plot(ax=ax, cmap=light_cmap, edgecolor='black', linewidth=0.75)

            for idx, row in africa.iterrows():
                country_name = row['name']
                if country_name in country_counts:
                    centroid = row['geometry'].centroid
                    label = f"{country_name}\n{country_counts[country_name]}"
                    ax.text(centroid.x, centroid.y, label, horizontalalignment='center', fontsize=6, color='k', weight="bold")

            plt.title('Number of Samples per Country in Africa')
            st.pyplot(fig)

elif st.session_state['page'] == 'resistance_mutations':
    st.subheader("Resistance Mutations")
    resistance_mutations = load_dataset_3()
    if resistance_mutations is not None:
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
    gff = load_gff_file()
    if gff is not None:
        final_dict = load_dataset_5()
        if final_dict is not None:
            resistance_mutations = load_dataset_3()
            if resistance_mutations is not None:
                drug_position = resistance_mutations.groupby('Drug')['Genomic position '].apply(list).to_dict()
                lineage_country = load_dataset_2()
                sample_ids = lineage_country['Name'].unique().tolist()
                selected_sample = st.selectbox("Select Sample ID", sample_ids)

                genome_length = 4411532

                def plot_original():
                    plt.figure(figsize=(50, 10))
                    for drug, positions in drug_position.items():
                        for pos in set(positions):
                            plt.axvline(x=pos, color='red', linewidth=5)
                    for i, feat in enumerate(gff.extract_features("gene")):
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

                for pos in set(selected_positions):
                    plt.axvline(x=int(pos), color='red')

                plt.xlim(0, genome_length)
                plt.ylim(0.75, 1.25)
                plt.yticks([])
                st.write(f"Mutations in: {selected_sample}")
                st.pyplot(plt)
                plt.close()
