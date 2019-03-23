#!/bin/bash
# A script for downloading several large, real, symmetric positive-definite
# matrices from the SuiteSparse Matrix Collection into a given folder.

if [ -z "${1}" ]; then
  echo "No download folder was supplied."
  exit 1
fi

# The directory to store the downloaded files in.
DOWNLOAD_FOLDER="${1}"

# The website containing the sparse matrix collection.
COLLECTION_SITE="https://sparse.tamu.edu/"

# A collection of matrices ordered from most-to-least nonzeros in the input.
declare -A MATRIX_NAMES
#MATRIX_NAMES["Queen_4147"]="Janna"
#MATRIX_NAMES["audikw_1"]="GHS_psdef"
MATRIX_NAMES["Serena"]="Janna"
MATRIX_NAMES["Geo_1438"]="Janna"
MATRIX_NAMES["Hook_1498"]="Janna"
MATRIX_NAMES["bone010"]="Oberwolfach"
MATRIX_NAMES["ldoor"]="GHS_psdef"
MATRIX_NAMES["boneS10"]="Oberwolfach"
MATRIX_NAMES["Emilia_923"]="Janna"
MATRIX_NAMES["PFlow_742"]="Janna"
MATRIX_NAMES["inline_1"]="GHS_psdef"
MATRIX_NAMES["nd24k"]="ND"
MATRIX_NAMES["Fault_639"]="Janna"
MATRIX_NAMES["StocF-1465"]="Janna"
MATRIX_NAMES["bundle_adj"]="Mazaheri"
MATRIX_NAMES["msdoor"]="INPRO"
MATRIX_NAMES["af_shell7"]="Schenk_AFE"
MATRIX_NAMES["af_shell8"]="Schenk_AFE"
MATRIX_NAMES["af_shell4"]="Schenk_AFE"
MATRIX_NAMES["af_shell3"]="Schenk_AFE"
MATRIX_NAMES["af_3_k101"]="Schenk_AFE"
# ...
MATRIX_NAMES["ted_B"]="Bindel"
MATRIX_NAMES["ted_B_unscaled"]="Bindel"
MATRIX_NAMES["bodyy6"]="Pothen"
MATRIX_NAMES["bodyy5"]="Pothen"
MATRIX_NAMES["aft01"]="Okunbor"
MATRIX_NAMES{"bodyy4"]="Pothen"
MATRIX_NAMES["bcsstk15"]="HB"
MATRIX_NAMES["crystm01"]="Boeing"
MATRIX_NAMES["nasa4704"]="Nasa"
MATRIX_NAMES["LF10000"]="Oberwolfach"
# ...
MATRIX_NAMES["mesh3e1"]="Pothen"
MATRIX_NAMES["bcsstm09"]="HB"
MATRIX_NAMES["bcsstm08"]="HB"
MATRIX_NAMES["nos1"]="HB"
MATRIX_NAMES["bcsstm19"]="HB"
MATRIX_NAMES["bcsstk22"]="HB"
MATRIX_NAMES["bcsstk03"]="HB"
MATRIX_NAMES["nos4"]="HB"
MATRIX_NAMES["bcsstm20"]="HB"
MATRIX_NAMES["bcsstm06"]="HB"
MATRIX_NAMES["bcsstk01"]="HB"
MATRIX_NAMES["mesh1em6"]="Pothen"
MATRIX_NAMES["mesh1em1"]="Pothen"
MATRIX_NAMES["mesh1e1"]="Pothen"
# ...

# Ensure that the directory to store the downloads into exists.
mkdir -p ${DOWNLOAD_FOLDER}

download_func () {
  for matrix_name in "${!MATRIX_NAMES[@]}"; do
    local group=${MATRIX_NAMES["${matrix_name}"]}
    local url="${COLLECTION_SITE}/MM/${group}/${matrix_name}.tar.gz"
    local tgz_dest="${DOWNLOAD_FOLDER}/${matrix_name}.tar.gz"
    local mtx_name="${matrix_name}.mtx"
    local prefixed_mtx_name="${matrix_name}/${mtx_name}"
    local final_dest="${DOWNLOAD_FOLDER}/${mtx_name}"

    # Skip this iteration of the loop if the final result already exists.
    if [ -e "${final_dest}" ]; then
      echo "${final_dest} already exists; skipping download and extraction."
      continue
    fi
    
    # We do not try to avoid redownloading if the tarball already exists, as
    # it might have been an aborted download.
    echo "Downloading ${url} to ${tgz_dest}."
    wget "${url}" -O "${tgz_dest}"

    # Extract the MTX file.
    echo "Extracting ${prefixed_mtx_name} to ${DOWNLOAD_FOLDER}."
    tar xzf "${tgz_dest}" -C "${DOWNLOAD_FOLDER}" "${prefixed_mtx_name}"

    # Move the MTX file into place.
    echo "Moving ${DOWNLOAD_FOLDER}/${prefixed_mtx_name} to ${final_dest}."
    mv "${DOWNLOAD_FOLDER}/${prefixed_mtx_name}" "${final_dest}"

    # Delete the tarball and the temporary directory.
    echo "Deleting ${tgz_dest}."
    rm -Rf "${tgz_dest}" "${DOWNLOAD_FOLDER}/${matrix_name}/"
  done
}

download_func
