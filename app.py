import gradio as gr
import pickle
import numpy as np

with open("yj.pkl","rb") as f:
    yj=pickle.load(f)
    
with open("best_model.pkl","rb") as f:
    model=pickle.load(f)
    
def pred(year,Adult_morality,infant_death,alcohol,hiv_aids,GDP,population,thinness_ten_to_nineteen,
            hepatitis,diphtheria,bmi,income_composition):
    predictions=model.predict([[
        year,np.ndarray.item(yj.transform(np.array(Adult_morality).reshape(1,-1))),np.ndarray.item(yj.transform(np.array(infant_death).reshape(1,-1))),
        np.ndarray.item(yj.transform(np.array(alcohol).reshape(1,-1))),np.ndarray.item(yj.transform(np.array(hiv_aids).reshape(1,-1))),np.ndarray.item(yj.transform(np.array(GDP).reshape(1,-1))),
        np.ndarray.item(yj.transform(np.array(population).reshape(1,-1))),np.ndarray.item(yj.transform(np.array(thinness_ten_to_nineteen).reshape(1,-1))),
        np.ndarray.item(yj.transform(np.array(hepatitis).reshape(1,-1))),np.ndarray.item(yj.transform(np.array(diphtheria).reshape(1,-1))),np.square(bmi),np.square(income_composition)
    ]])
    
    return predictions
    

year_input=gr.Number(label="Enter year from 2000 onwards")
Adult_morality_input=gr.Number(label="Probability of dying bw 15and 60 per 1000")
infant_death_input=gr.Number(label="Number of Infant Deaths")
alcohol_input=gr.Number(label="Alcohol Recorded per capita")
hiv_aids_input=gr.Number(label="Deaths per 1000 lives per HIV/AIDS (0-4 year olds)")
GDP_input=gr.Number(label="Gross Domestic Product per capita")
population_input=gr.Number(label="Population")
thinness_ten_to_nineteen_input=gr.Number(label="Prevalence of thinness in children from 10 to 19")
hepatitis_input=gr.Number(label="Hepatitis B immunization coverage in 1 year olds")
diphtheria_input=gr.Number(label="DTP3 immunization coverage among 1 year olds")
bmi_input=gr.Number(label="BMI")
income_composition_input=gr.Number(label="Human Development index in terms of income composition of resources")
output=gr.Number()

app=gr.Interface(
    fn=pred,
    inputs=[year_input,Adult_morality_input,infant_death_input,alcohol_input,hiv_aids_input,GDP_input,population_input,thinness_ten_to_nineteen_input,
            hepatitis_input,diphtheria_input,bmi_input,income_composition_input],
    outputs=output
)

app.launch()
