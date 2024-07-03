from astropy.io import fits
from glob import glob
import sys
import pandas as pd
from itertools import product
import os


# For unfold the first argument is the path of the folder containing the fits files to extract
# the second argument is the name of the directory where the csv files will be extracted
# the third argument must be the word "unfold"

# For regular save the first argument must be the path of the folder containing the fits files to extract
# the second argument is the name of the file to extract the data to

def get_name(path: str):
    try:
        place = path[::-1].index("/")
    except ValueError:
        place = path[::-1].index("\\")
    filename = path[-place:]
    return filename


def extract_rvc(path: str) -> dict:
    files: list = glob(path + r'*.fits')
    velocities: list = []
    julian_days: list = []
    ccf_noise: list = []
    targets: list = []
    instrument = []
    filenames = []
    for file in files:
        filenames.append(get_name(file))
        with fits.open(file) as hdu:
            header = hdu[0].header
            velocity = header['ESO DRS CCF RVC']
            julian = header['ESO DRS BJD']
            noise = header['ESO DRS CCF NOISE']
            target = header['ESO OBS TARG NAME']
            instr = header["INSTRUME"]

            velocities.append(velocity)
            julian_days.append(julian)
            ccf_noise.append(noise)
            targets.append(target)
            instrument.append(instr)
    return {'Target': targets, 'DRS.CCF.RVC': velocities, 'DRS.BJD': julian_days, 'DRS.CCF.NOISE': ccf_noise,
            "Instrument": instrument, "Filename": filenames}


def save_data(data, dest) -> None:
    pd.DataFrame(data).to_csv(dest, index=False)


def unfold(dats, folder: str):
    df = pd.DataFrame(data=dats)
    target_list = df["Target"].unique()
    instrument_list = df["Instrument"].unique()
    os.mkdir(folder)
    os.chdir(folder)
    for x in product(target_list, instrument_list):
        create = x[0] + "_" + x[1] + ".csv"
        data = df[(df["Target"] == x[0]) & (df["Instrument"] == x[1])]
        save_data(data=data, dest=create)


def main() -> None:
    data = extract_rvc(sys.argv[1])
    try:
        unf = sys.argv[3]
        if unf == "unfold":
            unfold(data, sys.argv[2])
        else:
            raise ValueError('Third argument must be "unfold" if provided')
    except IndexError:
        save_data(data, sys.argv[2])


if __name__ == '__main__':
    main()
