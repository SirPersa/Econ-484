import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Country Name": [
        "Afghanistan", "Andorra", "Australia", "Chile", "Czechia", "Georgia", "Italy", "Jamaica", "Jordan", "Kazakhstan",
        "Kenya", "Kyrgyzstan", "Libya", "Malaysia", "Maldives", "Moldova", "Mongolia", "Norway", "Philippines", "Qatar",
        "Senegal", "Slovenia", "Somalia", "South Africa", "Sudan", "Suriname", "Switzerland", "Turkey", "Yemen", "Zambia"
    ],
    "Global Rent Price Index (5 year %)": [
        12.5, 18.19, 17.03, 25.71, 27.59, 22.01, 8.17, 15.8, 18.5, 23.4,
        14.2, 19.8, 28.01, 11.3, 20.5, 16.9, 24.6, 14.83, 13.66, 17.5,
        13.1, 46.58, 15.01, 11.53, 35.01, 30.5, 9.02, 639.29, 25.01, 14.75
    ],
    "Home Ownership Rate": [
        75, 78, 66.3, 65, 76, 80, 75.9, 70, 68, 98,
        75, 98, 70, 76.9, 60, 90, 75, 79.2, 55, 45,
        65, 75.2, 65, 69.7, 75, 68, 42.3, 56.7, 80, 60
    ],
    "Avarage Housing Prices (USD/m^2)": [
        700, 5000, 6500, 2500, 3500, 1700, 3500, 2000, 1400, 1150,
        1500, 900, 500, 2000, 4000, 950, 1400, 7500, 1850, 3250,
        1150, 3000, 350, 1500, 250, 1150, 11500, 1000, 275, 900
    ],
    "Share of construction in GDP": [
        6.5, 8.9, 7.2, 7.5, 5.3, 8.4, 5.7, 8.1, 6.3, 6.1,
        7.8, 9.7, 4.5, 4.8, 12.1, 9.5, 10.2, 5.1, 9.1, 11.5,
        6.5, 6.8, 3.1, 3.5, 2.8, 5.5, 5.0, 6.7, 2.5, 8.3
    ],
    "Motor Vehicle Production": [
        0, 0, 7522, 5500, 1458892, 0, 591067, 0, 0, 144624,
        1135, 0, 0, 790347, 0, 0, 0, 0, 126571, 0,
        0, 0, 0, 600, 0, 0, 0, 1365296, 0, 0
    ],
    "Motor Vehicles Per Capita (1000 People)": [
        55, 121, 776, 312, 664, 441, 753, 203, 173, 250,
        62, 200, 463, 678, 250, 323, 347, 629, 122, 512,
        32, 581, 4.5, 208, 30, 394, 603, 354, 122, 45
    ],
    "Vehicle Exports (USD thousand)": [
        3031, 13619, 1597669, 312176, 54050740, 14529, 47008486, 10006, 23058, 365836,
        90032, 2121, 1031, 2440313, 777, 54125, 17708, 1384806, 1012996, 7189,
        54323, 10880246, 4221, 12570385, 368, 3389, 2984995, 31945201, 247, 62416
    ]
}

df = pd.DataFrame(data)

real_estate_cols = [
    'Global Rent Price Index (5 year %)',
    'Home Ownership Rate',
    'Avarage Housing Prices (USD/m^2)',
    'Share of construction in GDP'
]
automobile_cols = [
    'Motor Vehicle Production',
    'Motor Vehicles Per Capita (1000 People)',
    'Vehicle Exports (USD thousand)'
]

scaler = MinMaxScaler()
df[real_estate_cols + automobile_cols] = scaler.fit_transform(df[real_estate_cols + automobile_cols])

df["RealEstateScore"] = df[real_estate_cols].sum(axis=1)
df["AutomobileScore"] = df[automobile_cols].sum(axis=1)
df["CountryType"] = df.apply(
    lambda row: "Real Estate Country" if row["RealEstateScore"] > row["AutomobileScore"] else "Automobile Country",
    axis=1
)

# Yeni ülke tahmini
new_country = {
    'Country Name': "Çağrının Ülkesi",
    'Global Rent Price Index (5 year %)': 20,
    'Home Ownership Rate': 70,
    'Avarage Housing Prices (USD/m^2)': 2000,
    'Share of construction in GDP': 7,
    'Motor Vehicle Production': 100000,
    'Motor Vehicles Per Capita (1000 People)': 350,
    'Vehicle Exports (USD thousand)': 500000
}

country_name = new_country['Country Name']
metrics_only = {k: v for k, v in new_country.items() if k != 'Country Name'}
new_df = pd.DataFrame([metrics_only])
normalized = scaler.transform(new_df)
real_estate_score = normalized[0][:4].sum()
automobile_score = normalized[0][4:].sum()
country_type = "Real Estate Country" if real_estate_score > automobile_score else "Automobile Country"

print("\n=== Prediction for a New Country ===")
print(f"Country: {country_name}")
print(f"Real Estate Score: {real_estate_score:.3f}")
print(f"Automobile Score: {automobile_score:.3f}")
print(f"Predicted Type: {country_type}")


score_df = df[['Country Name', 'RealEstateScore', 'AutomobileScore']].copy()
score_long = pd.melt(score_df, id_vars='Country Name', var_name='Category', value_name='Score')

plt.figure(figsize=(14, 10))
sns.barplot(data=score_long, y='Country Name', x='Score', hue='Category')
plt.title("Real Estate and Automobile Scores by Country")
plt.xlabel("Score (Normalized)")
plt.ylabel("Country")
plt.legend(title="Category")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
df['CountryType'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140,
                                      colors=['#66c2a5', '#fc8d62'])
plt.title("Distribution of Country Types")
plt.ylabel("")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="RealEstateScore", y="AutomobileScore", hue="CountryType", s=100)
plt.plot([0, 4], [0, 4], 'k--', linewidth=1)
plt.title("Score Comparison of Countries")
plt.xlabel("Real Estate Score")
plt.ylabel("Automobile Score")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n Top 5 Countries by Real Estate Score:")
print(df.sort_values("RealEstateScore", ascending=False)[["Country Name", "RealEstateScore", "CountryType"]].head(5))

print("\n Top 5 Countries by Automobile Score:")
print(df.sort_values("AutomobileScore", ascending=False)[["Country Name", "AutomobileScore", "CountryType"]].head(5))
