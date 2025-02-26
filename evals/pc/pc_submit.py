from selenium import webdriver
from time import sleep
import pandas as pd
import argparse


def submit_page(page_no,beg,end,row,driver):
  xpath='/html/body/div[2]/div[2]/main/article/form/span['
  for i in range(beg,end+1):
    curr_x = xpath + f'{i-beg+1}]/fieldset/div/div/div'
    # print(curr_x)
    if(row[f'answer_{i}']==0):
      curr_x = curr_x + '/label[0]/span/input'
      driver.find_element(By.XPATH,curr_x).click()
    elif(row[f'answer_{i}']==1):
      curr_x = curr_x + '/label[1]/span/input'
      driver.find_element(By.XPATH,curr_x).click()
    elif(row[f'answer_{i}']==2):
      curr_x = curr_x + '/label[2]/span/input'
      driver.find_element(By.XPATH,curr_x).click()
    elif(row[f'answer_{i}']==3):
      curr_x = curr_x + '/label[3]/span/input'
      driver.find_element(By.XPATH,curr_x).click()
    else:
      curr_x = curr_x + '/label[2]/span/input'
      driver.find_element(By.XPATH,curr_x).click()
      print(row,i)
  driver.find_element(By.XPATH,'/html/body/div[2]/div[2]/main/article/form/button').click()
  sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Process CSV files for scoring.")
    parser.add_argument("--key", type=str, help="OpenAI API key")
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        required=True,
        help="The path for the input CSV file.",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        required=True,
        help="The path for the new CSV output file.",
    )  
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    driver = webdriver.Chrome()

    column_names = ['iteration', 'step','ec','soc']
    out_df = pd.DataFrame(columns=column_names)
    for i in range(df.shape[0]):
        row = df.iloc[i]
        new_row = {}
        driver.get(url=f'https://www.politicalcompass.org/test/en?page=1')

        submit_page(1,1,7,row,driver)
        submit_page(2,8,21,row,driver)
        submit_page(3,22,39,row,driver)
        submit_page(4,40,51,row,driver)
        submit_page(5,52,56,row,driver)
        submit_page(6,57,62,row,driver)

        curr_url = driver.current_url
        ec = curr_url.split('=')[-2].split('&')[0]
        soc = curr_url.split('=')[-1]
        new_row['ec']=ec
        new_row['soc']=soc
        new_row['iteration'] = df['iteration'][i]
        new_row['step'] = df['step'][i] 
        out_df.loc[len(out_df)] = new_row
    
    out_df.to_csv(args.output)
    print(f"Saved results to {args.output}.")


if __name__ == "__main__":
    main()
