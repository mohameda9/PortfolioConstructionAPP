from app.services.common_imports import *


def get_ff_data():
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    response = requests.get(ff_url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    with zip_file.open(zip_file.namelist()[0]) as csvfile:
        ff_factors = pd.read_csv(csvfile, skiprows=3)

    ff_factors.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    ff_factors["date"] = (
        ff_factors["date"]
        .astype(str)
        .apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    )
    ff_factors.set_index("date", inplace=True)
    risk_free_data = ff_factors[["RF"]].apply(
        lambda x: round((1 + x) ** (1 / 365) - 1, 8)
    )  #### might need to update to t-1 logic

    ff_factors.drop(columns=["RF"], inplace=True)
    ff_factors /= 100

    risk_free_data
    return risk_free_data, ff_factors
