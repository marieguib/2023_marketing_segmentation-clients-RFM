### 0. Initialisation du projet ---
import datetime as datetime
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# import csv
pd.set_option("display.max_columns",500)

R_COMPLEMENT_INDIVIDU_2016 = pd.read_csv('R_COMPLEMENT_INDIVIDU_2016.csv',delimiter=";",encoding="iso-8859-1")
R_INDIVIDU_2016 = pd.read_csv('R_INDIVIDU_2016.csv',delimiter=";",encoding="iso-8859-1",dtype={"DATE_NAISS_A":'str',"DATE_NAISS_M":'str',"DATE_NAISS_J":'str'})
R_MAGASIN = pd.read_csv('R_MAGASIN.csv',delimiter=";",encoding="iso-8859-1")
R_REFERENTIEL = pd.read_csv('R_REFERENTIEL.csv',delimiter=";",encoding="iso-8859-1",dtype={"EAN":"str"})
R_TICKETS_2016 = pd.read_csv('R_TICKETS_2016.csv',delimiter=";",encoding="iso-8859-1",dtype={"EAN":"str"})
# print(R_TICKETS_2016.shape)
# 1 ligne correspond à 1 produit acheté par un client à une date données
R_TYPO_PRODUIT = pd.read_csv('R_TYPO_PRODUIT.csv',delimiter=";",encoding="iso-8859-1")



##################################################################################################################################
# -------------- 1. Construction de la table finale permettant l’analyse des tickets et l’analyse des clients --------------------
##################################################################################################################################
# Objectif : construire table finale pour faire la segmentation RFM


# ----------------- Etape 1 – Construction de la table au niveau Individus ---

# On doit retrouver 36 158 individus
# Il faut conserver les individus de la table R_INDIVIDU_2016 n’ayant pas de complément
# La clé de jointure entre les deux tables est ID_INDIVIDU

R_INDIVIDU = pd.merge(left=R_INDIVIDU_2016,right=R_COMPLEMENT_INDIVIDU_2016,how="left",on="ID_INDIVIDU")
# print(R_INDIVIDU.shape)
# print(R_INDIVIDU_2016.shape)

# Renommer code_magasin
R_INDIVIDU=R_INDIVIDU.rename(columns={'CODE_MAGASIN':"MAGASIN_GESTIONNAIRE"})
# print(R_INDIVIDU.shape)

# Quel filtre à appliquer ?
# On prend les tickets d'achats entre 2014 et 2016
# On pourrait filtrer les individus qui n'étaient pas encore là sur la période d'analyse 
# => Tout client après 2014 n'est pas utile : 
# on ne peut pas comparer un client avec 10 ans d'ancienneté et un avec 4 mois => période d'analyse (2 ans max) plus courte 
# => forcément moins d'achats 
# On ne garde que les clients avec une création de carte inférieure à 2014
# Sur chaque client on aura analysé 2 ans de tickets

date_debut = datetime.datetime(2014,9,1)
R_INDIVIDU["DATE_CREATION_CARTE"] = pd.to_datetime(R_INDIVIDU['DATE_CREATION_CARTE'],format="%d/%m/%Y",errors="ignore")
R_INDIVIDU = R_INDIVIDU.loc[R_INDIVIDU["DATE_CREATION_CARTE"] < date_debut] 
# print(R_INDIVIDU.shape)

# - Validation
# R_INDIVIDU['DATE_CREATION_CARTE'].describe() # permet d'avoir les statistiques de base



# ----------------- Etape 2 – Calcul Age / Ancienneté sur la table « Individus » --------------------

date_fin_etude = datetime.datetime(2016,8,31)

# - Calcul de l'âge du client : date de naissance - date d'extraction
# Unité pour l'âge : année
# - Création d'une colonne daite_naissance en regroupant les trois colonnes 
annee = R_INDIVIDU['DATE_NAISS_A']
mois = R_INDIVIDU['DATE_NAISS_M']
jour = R_INDIVIDU['DATE_NAISS_J']
date_naiss = annee + "/" + mois + "/" + jour
date_naiss = pd.to_datetime(date_naiss)
R_INDIVIDU["date_naissance"] = date_naiss
# - Calcul de l'âge
R_INDIVIDU["age"] = (date_fin_etude - date_naiss)//datetime.timedelta(days=365)
# - Nettoyage de l'âge
R_INDIVIDU["age"] = R_INDIVIDU["age"].apply(lambda x : x if x>=15 and x<=90 else None)
# print(R_INDIVIDU["age"].describe())


# - Calcul de l'ancienneté : date de création de la carte - date extraction -- > unité : mois
R_INDIVIDU["anciennete"] = (date_fin_etude-R_INDIVIDU["DATE_CREATION_CARTE"])//datetime.timedelta(days=30)

# - Nettoyage de l'ancienneté
# Données aberrantes : jour > 31 ; mois > 12 ; année > 2016
# Client qui a 200 ans : on remplace par des données manquantes
anciennete_min = 365*2
R_INDIVIDU["anciennete"] = R_INDIVIDU["anciennete"].apply(lambda x : 
                                                x
                                                if x <= 120
                                                else None)

df_tmp=R_INDIVIDU.sort_values(by='anciennete', ascending=True)
# print(R_INDIVIDU["anciennete"].describe())



#----------------- Etape 3 - Travail sur la table des Tickets : Sélection du bon périmètre sur la table ticket --------------

# - Filtre des tickets
R_TICKETS_2016["DATE_ACHAT"] = pd.to_datetime(R_TICKETS_2016["DATE_ACHAT"],format="%d/%m/%Y",errors="ignore")
# Les données doivent avoir 2 ans d'historique : elles démarrent à la date de fin d'étude et remontent à date_fin_etude-2ans

borne_inf  = datetime.datetime(2014,9,1)
borne_sup = datetime.datetime(2016,8,31)
R_TICKETS_2016 = R_TICKETS_2016.loc[(R_TICKETS_2016["DATE_ACHAT"] >= borne_inf) & (R_TICKETS_2016["DATE_ACHAT"] <= borne_sup)] 

# PERIMETRE DE L'ANALYSE
# print(R_TICKETS_2016["DATE_ACHAT"].describe())
# print(R_TICKETS_2016.shape)




#----------------- Etape 4 - Enrichissement de la table des TICKETS ----------------------------
# - Correction

TICKETS_MAG=pd.merge(R_TICKETS_2016
                     , R_MAGASIN[['CODE_BOUTIQUE','REGIONS','CENTRE_VILLE','TYPE_MAGASIN','REGIONS_COMMERCIAL']]
                     , on ='CODE_BOUTIQUE'
                     , how='left')
# TICKETS_MAG.shape

# - JOINTURE AVEC REFERENTIEL
TICKETS_MAG_REF = pd.merge(TICKETS_MAG,R_REFERENTIEL[['EAN','MODELE']],on='EAN',how='left')
# TICKETS_MAG_REF.shape
# TICKETS_MAG_REF.head()

# - JOINTURE AVEC PRODUIT
R_MATRICE_TRAVAIL = pd.merge(TICKETS_MAG_REF, R_TYPO_PRODUIT[['MODELE','Ligne','Famille']]
                             ,on='MODELE',how='left')
# R_MATRICE_TRAVAIL.shape
# print(R_MATRICE_TRAVAIL.head())



# --------------------- Partie 2 : Auditer la table r_Matrice_travail et r_individu_OK ------------------------

# ---------- Etape 1 – Analyse de la table r_Matrice_travail---------------------------

# Prendre des exemples / analyser les données / données aberrantes / nettoyage de la donnée
# print(R_MATRICE_TRAVAIL.info())

# for i in ["REGIONS","CENTRE_VILLE","TYPE_MAGASIN","REGIONS_COMMERCIAL","MODELE","Ligne","Famille"]:
#     print(f'Variable {i} \n')
#     print(R_MATRICE_TRAVAIL[i].value_counts())
#     print("######################\n")


# Constat : 
# - PRIX_AP_REMISE : on des valeurs égales à 0 => cadeaux 
# - REMISE : possède des valeurs négatives, on décide donc de tout passer en valeur absolue
# - REMISE_VALEUR : est-ce normal d'avoir 1007% de remise ?
# - Des variables présentent des valeurs similaires qu'on va fusionner

# --- Modification de la colonne CENTRE_VILLE
# print(R_MATRICE_TRAVAIL["CENTRE_VILLE"].unique())
# On remarque deux labels pour identifier les centres commercials : centre commercial et centre co

R_MATRICE_TRAVAIL["CENTRE_VILLE"] = R_MATRICE_TRAVAIL["CENTRE_VILLE"].apply(

    lambda x : f'Centre Commercial'
        if x.lower() =='centre co'
        else x)

# print(R_MATRICE_TRAVAIL["CENTRE_VILLE"].unique())

# --- Modification de la colonne TYPE_MAGASIN

#la principale différence entre un magasin en succursale  et un magasin affilié est que le premier est directement possédé et géré par l'entreprise mère, tandis que le second est exploité par un propriétaire indépendant qui utilise la marque et le système de franchise de l'entreprise mère.
# Autrement dit les magasins propres sont aussi des succursale, on va transformer les magasins codé en mag propre en succursale

R_MATRICE_TRAVAIL["TYPE_MAGASIN"] = R_MATRICE_TRAVAIL["TYPE_MAGASIN"].apply(

    lambda x : f'Succursale'
        if x.lower() =='mag propre'
        else x)

# print(R_MATRICE_TRAVAIL["TYPE_MAGASIN"].unique())

# -- Modification de la colonne REMISE_VALEUR
masque1 = R_MATRICE_TRAVAIL["REMISE_VALEUR"]==1007
# print(R_MATRICE_TRAVAIL.loc[masque1])
# On observe un pourcentage de remise de 1007 donc on remplace cette valeur par une valeur manquante :
R_MATRICE_TRAVAIL.loc[R_MATRICE_TRAVAIL["ID_INDIVIDU"]==7834, "REMISE_VALEUR"] = None


# -- Modification de la colonne REMISE
# On passe en valeur absolue les remises inférieures à 0 pour homgénéiser la base de données
masque2 = R_MATRICE_TRAVAIL["REMISE"]<0
R_MATRICE_TRAVAIL.loc[masque2, "REMISE"] = abs(R_MATRICE_TRAVAIL.loc[masque2, "REMISE"]) 
# print(R_MATRICE_TRAVAIL.loc[masque2])

# --- Modification de la colonne MODELE

# print(R_MATRICE_TRAVAIL["MODELE"].unique())
# On remarque des modalités identiques avec des noms différents : 
#    - FAV0 et FAVORI
#    - DIVE et DIVERS

R_MATRICE_TRAVAIL["MODELE"] = R_MATRICE_TRAVAIL["MODELE"].apply(

    lambda x : f'FAVORI'
        if x =='FAVO'
        else x)

# Favo ce sont des cadeaux

R_MATRICE_TRAVAIL["MODELE"] = R_MATRICE_TRAVAIL["MODELE"].apply(

    lambda x : f'DIVERS'
        if x =='DIVE'
        else x)


# print(R_MATRICE_TRAVAIL["MODELE"].unique())



# --------------------- Partie 3 : Calculer avec la table Matrice_travail_OK contenant les modifications précédentes ------------------------

# ---- ---------- Etape 1 –  Déterminer la règle permettant d’identifier un ticket d’achat unique une visite en magasin---------------------------

# - Extraction de l'individu 174591
Ticket_174591 = R_MATRICE_TRAVAIL.loc[R_MATRICE_TRAVAIL["ID_INDIVIDU"]==783, :]
# print(Ticket_174591)

# Pour identifier un ticket unique on a besoin de :  NUM_TICKET / DATE_ACHAT / ID_INDIVIDU / CODE_BOUTIQUE

gb_MATRICE_TRAVAIL = R_MATRICE_TRAVAIL.groupby(["ID_INDIVIDU","CODE_BOUTIQUE","DATE_ACHAT","NUM_TICKET"],as_index=False) # as.index = FALSE -> reste des colonnes et ne devient plus des index




# ---- ---------- Etape 2 –  Déterminer la règle permettant d’identifier un ticket d’achat unique une visite en magasin---------------------------

# -- Statistiques par visite
df_visite = gb_MATRICE_TRAVAIL.agg({"QUANTITE":sum,"PRIX_AP_REMISE":sum})
df_visite = df_visite.rename(columns={"QUANTITE":"NB_PRODUITS","PRIX_AP_REMISE":"CA_VISITE"})

# - Calcul du prix moyen
df_visite["PRIX_MOYEN"]=df_visite["CA_VISITE"]/df_visite["NB_PRODUITS"]
# print(df_visite.head)

# - Validation
# df_4 = R_MATRICE_TRAVAIL.loc[R_MATRICE_TRAVAIL['ID_INDIVIDU']==4]
# df_visite_4 = df_visite.loc[df_visite['ID_INDIVDU']==4]


# --- Statistiques par individu

# correction
r_Indicateur_achats = df_visite.groupby("ID_INDIVIDU",as_index=False).agg(
    MONTANT_CUMULE = ("CA_VISITE","sum"),
    NB_VISITE = ("CA_VISITE","size"),
    CA_MOY_VISITE = ("CA_VISITE","mean"),
    NB_PRDT_MOY_VISITE = ('NB_PRODUITS',"mean")
)

# print(df_indiv)



# --- Indicateurs par individu:
# - RECENCE ACHAT
r_recence = df_visite.groupby("ID_INDIVIDU",as_index=False).agg(
    date_plus_recente = ("DATE_ACHAT",lambda x : max(pd.to_datetime(x,format="%d/%m/%Y")))
)

r_recence["RECENCE"] = (pd.to_datetime(date_fin_etude,format = '%Y-%m-%d')-r_recence["date_plus_recente"]) /pd.Timedelta(days=1)
r_recence = r_recence[["ID_INDIVIDU",'RECENCE']]


# - NB MAGASIN DIFF : fait au-dessus

# - Nb Cadeaux reçus : a faire avec favoris
R_MATRICE_TRAVAIL["nb_cadeau"] = R_MATRICE_TRAVAIL.apply(lambda x : x['QUANTITE'] if x["MODELE"] == 'FAVORI' else 0, axis = 1)
r_indicateur_suplementaire = R_MATRICE_TRAVAIL.groupby('ID_INDIVIDU',as_index=False).agg(
    NB_MAG_DIFF = ("CODE_BOUTIQUE",'nunique'),
    NB_LIGNE_DIFF = ("Ligne","nunique"),
    NB_FAM_DIFF = ("Famille","nunique"),
    NB_CADEAU = ('nb_cadeau',"sum")
)


# - Part des visites dans le magasin gestionnaire PART_VIST_MAG_GEST = Nb visites MAG_GESTIONNAIRE / Nb visites)
part_mag_gestion = df_visite.merge(
    R_INDIVIDU[["ID_INDIVIDU","MAGASIN_GESTIONNAIRE"]],
    on = "ID_INDIVIDU", how="left"
)

# - Création d'une variable = 1 si achat dans le magasin gestionnaire

part_mag_gestion['Top_mag2'] = part_mag_gestion.apply(lambda x : 1 if x['MAGASIN_GESTIONNAIRE']==x['CODE_BOUTIQUE'] else 0, axis=1)
part_mag_gestion['Top_mag'] = np.where(part_mag_gestion['MAGASIN_GESTIONNAIRE'] == part_mag_gestion['CODE_BOUTIQUE'], 1,0)


# Calcul de la part des achats dans le magasin gestionnaire
part_mag_gestion = part_mag_gestion.groupby('ID_INDIVIDU',as_index=False).agg(
    VIS_MAG_GESTION=('Top_mag', 'sum'), N_VISITE=('MAGASIN_GESTIONNAIRE','size')
)
part_mag_gestion['PART_VIST_MAG_GEST']=part_mag_gestion['VIS_MAG_GESTION'].astype(float)/part_mag_gestion['N_VISITE']

# Filtre pour l'individu 1396
# print(part_mag_gestion[part_mag_gestion['ID_INDIVIDU'] == 1396])


# --- Regroupement r_matrice_finale

# Sélection des colonnes du dataframe r_individu_OK
r_Matrice_finale = pd.DataFrame(R_INDIVIDU, columns=['ID_INDIVIDU', 'age'
                    , 'anciennete', 'MAGASIN_GESTIONNAIRE', 'SEXE', 'CIVILITE'])

# Jointure avec R_MAGASIN
r_Magasin = pd.DataFrame(R_MAGASIN, columns=['CODE_BOUTIQUE', 'REGIONS', 'CENTRE_VILLE', 'TYPE_MAGASIN','MER_TERRE'])
r_Matrice_finale = pd.merge(r_Matrice_finale, r_Magasin, how='left', left_on='MAGASIN_GESTIONNAIRE', right_on='CODE_BOUTIQUE')

# Jointure avec r_Indicateurs_achats
r_Matrice_finale = pd.merge(r_Matrice_finale, r_Indicateur_achats, how='left', on='ID_INDIVIDU')

# Jointure avec r_Recence
r_Matrice_finale = pd.merge(r_Matrice_finale, r_recence, how='left', on='ID_INDIVIDU')

# Jointure avec r_Indicateurs_supplementaire
r_Matrice_finale = pd.merge(r_Matrice_finale, r_indicateur_suplementaire, how='left', on='ID_INDIVIDU')

# Jointure avec Part_mag_gestion
r_Matrice_finale = pd.merge(r_Matrice_finale, part_mag_gestion, how='left', on='ID_INDIVIDU')

# Comptage des observations pour chaque valeur unique de CENTRE_VILLE
count_CENTRE_VILLE = r_Matrice_finale.groupby('CENTRE_VILLE').size().reset_index(name='count')

client_inactif=r_Matrice_finale[r_Matrice_finale['NB_VISITE'].isnull()]
rfm=r_Matrice_finale[r_Matrice_finale['NB_VISITE'].isnull()==False]





##########################################################################################################
# ---------------------------------  Partie 4 : Constitution de la RFM ------------------------------------
###########################################################################################################


# --> découper les variables (récence, fréquence, montant) avec des seuils 1/3 1/3 1/3
# --> ajout d'une colonne pour chaque variable (récence avec valeurs faible/moyen/faible)
# --> on regroupe fréquence / montant : création d'une colonne 
# --> on croise récence fréquence et montant : RFM1 à RFM9
# --> création d'une colonne segment RFM (petit client / grand consommateur)

# - Couper la colonne RECENCE en 3 catégories
rfm['RECENCE_MOD'] = pd.cut(rfm['RECENCE'], np.quantile(rfm['RECENCE'], q = np.arange(0,1.01,1/3)), labels=["faible","moyen","fort"],include_lowest=True)
# print(rfm['RECENCE_MOD'].describe())

# - Couper la colonne NB_PRDT_MOY_VISITE en 3 catégories
rfm['FREQUENCE'] = pd.cut(rfm['NB_VISITE'], np.quantile(rfm['NB_VISITE'], q = np.arange(0,1.01,1/3)), labels=["faible","moyen","fort"],include_lowest=True)
# print(rfm['FREQUENCE'].describe())

# - Couper la colonne MONTANT_CUMULE en 3 catégories
rfm['MONTANT'] = pd.cut(rfm['MONTANT_CUMULE'], bins=np.quantile(rfm['MONTANT_CUMULE'], q = np.arange(0,1.01,1/3)), labels=["faible","moyen","fort"],include_lowest=True)
# print(rfm['MONTANT'].describe())
# print(rfm.head)


rfm["FM"] =  rfm.apply(lambda x : f'FM_faible'
                       if (x['FREQUENCE'] == 'faible' and x['MONTANT'] == 'faible') 
                            or (x['FREQUENCE'] == 'faible' and x['MONTANT'] == 'moyen') 
                            or (x['FREQUENCE'] == 'moyen' and x['MONTANT'] == 'faible')
                       else  
                        f'FM_moyen'
                         if (x['FREQUENCE'] == 'faible' and x['MONTANT'] == 'fort') 
                            or (x['FREQUENCE'] == 'moyen' and x['MONTANT'] == 'moyen') 
                            or (x['FREQUENCE'] == 'fort' and x['MONTANT'] == 'faible')
                        else f'FM_fort',
                        axis = 1)


# Validation 
# print(rfm["FM"].describe())
# print(pd.crosstab(rfm["FREQUENCE"],rfm["MONTANT"]))

rfm["RFM"] =rfm.apply(lambda x : 'RFM1'
                       if (x['FM'] == 'FM_faible' and x['RECENCE_MOD'] == 'fort') 
                       else 'RFM2' if (x['FM'] == 'FM_faible' and x['RECENCE_MOD'] == 'moyen') 
                       else  'RFM3' if (x['FM'] == 'FM_faible' and x['RECENCE_MOD'] == 'faible') 
                       else 'RFM4' if (x['FM'] == 'FM_moyen' and x['RECENCE_MOD'] == 'fort') 
                       else 'RFM5' if (x['FM'] == 'FM_moyen' and x['RECENCE_MOD'] == 'moyen') 
                       else  'RFM6' if (x['FM'] == 'FM_moyen' and x['RECENCE_MOD'] == 'faible') 
                       else 'RFM7' if (x['FM'] == 'FM_fort' and x['RECENCE_MOD'] == 'fort')
                       else 'RFM8' if (x['FM'] == 'FM_fort' and x['RECENCE_MOD'] == 'moyen')
                       else 'RFM9', 
                       axis = 1)

# print(rfm["RFM"].describe())

# Construction des segments
rfm["SEGMENTv2"] = rfm.apply(lambda x : 'Weaks'
                       if (x['RFM'] == 'RFM1') 
                       else 'Lowest-spending customers' if ((x['RFM'] == 'RFM2') or (x['RFM'] == 'RFM3'))  
                       else 'Churns' if ((x['RFM'] == 'RFM4') or (x['RFM'] == 'RFM7'))
                       else 'Good customers' if ((x['RFM'] == 'RFM5') or (x['RFM'] == 'RFM6'))
                       else 'Best customers',
                       axis = 1)

# print(rfm["SEGMENTv2"].value_counts())

# print(rfm.head)




##################################################################################################################################
# --------------------------------------- Graphiques et analyses des bases de données  -------------------------------------------
##################################################################################################################################


# CONTEXTE ------------------------------------------------------------------------------

# Nombre de magasins initial : 
# print(R_MAGASIN['CODE_BOUTIQUE'].nunique())
# Nombre de magasins final : 
# print(R_MATRICE_TRAVAIL['CODE_BOUTIQUE'].nunique())


# Number of customers by segment

customers_seg = rfm.groupby("SEGMENTv2",as_index=False).size()
# print(customers_seg)
# customers_seg.to_excel('customers_seg.xlsx', index=False)

# Graphique individus ----------------------
# print(R_INDIVIDU.columns)
# print(R_INDIVIDU.age)

# Médiane de l'âge 
mediane_age = R_INDIVIDU['age'].median()
# print(mediane_age)


# Moyenne de l'âge 
moy_age = R_INDIVIDU['age'].mean()
# print(moy_age)

# Mode de l'âge 
mode_age = R_INDIVIDU['age'].mode()
# print("le mode est :",mode_age[2])


# Graphique 
plt.hist(R_INDIVIDU['age'],bins = 7,color="#87CEFA")
plt.axvline(mediane_age,color="r",linestyle = "dashed",linewidth =1)
plt.axvline(moy_age,color="blue",linestyle = "dashed",linewidth =1)
plt.annotate("Median age",xy=(45,6500),xytext = (25,4000),arrowprops = {"width":3, "headwidth" :10,"headlength":10,"edgecolor":'red',"facecolor":'red'},horizontalalignment="center",fontsize=10,color='r')
plt.annotate("Mean age",xy=(47,7700),xytext = (67,8000),arrowprops = {"width":3, "headwidth" :10,"headlength":10,"edgecolor":'blue',"facecolor":'blue'},horizontalalignment="center",fontsize=10,color='blue')
plt.title('Distribution of customers by age group',fontweight = 'bold')
plt.xlabel('Age group',fontweight = 'bold')
plt.ylabel('Number of customers',fontweight = 'bold')
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()



# --------------------------------------------------------------------
# Graphique dimension produit ----------------------------

# print(rfm["TYPE_MAGASIN"].describe())
rfm["TYPE_MAGASIN"] = rfm["TYPE_MAGASIN"].apply(

    lambda x : f'Succursale'
        if x.lower() =='mag propre'
        else x)


# # Graphique : Average amount per customers type
g2 = rfm.groupby('SEGMENTv2',as_index=False)['MONTANT_CUMULE'].mean()
graph2 = plt.bar(g2.SEGMENTv2, g2.MONTANT_CUMULE,color = ['steelblue', 'lightsteelblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue'])
for rect in graph2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
plt.xlabel("Customers type",fontweight = 'bold')
plt.ylabel("Average amount (in €)",fontweight = 'bold')
plt.title('Average amount per customers type',fontweight = 'bold')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.show()



# # Tableau résumé 

# tab_resume = rfm.groupby("RFM",as_index=False).agg(
#     effectif = ("ID_INDIVIDU","size"),
#     recence_moyenne = ("RECENCE","mean"),
#     montant_moyen = ("MONTANT_CUMULE","mean")
# )

# tab_resume["recence_moyenne"] = round(tab_resume["recence_moyenne"])
# tab_resume["montant_moyen"] = round(tab_resume["montant_moyen"])
# tab_resume["pourc_effectif"] = round(tab_resume["effectif"]/100)
# # print(tab_resume)
# tab_resume.to_excel('tab_resume.xlsx', index=False)



# -- Graphique average number of customers type for each modality "Nombre de magasin"

g5 = rfm.groupby(['NB_MAG_DIFF','SEGMENTv2'])["ID_INDIVIDU"].size().unstack()
# print(g5)
ax = g5.plot(kind='bar',color = ['steelblue', 'lightsteelblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue'])
for p in ax.containers:
    ax.bar_label(p, label_type='edge', fontsize=10)
plt.xlabel("Number of stores",fontweight = 'bold')
plt.ylabel("Number of customers",fontweight = 'bold')
plt.xlim(0,5)
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.title('The average number of customers for each category of the variable "Number of Stores"',fontweight = 'bold')
plt.legend(title = "Customers type",loc='upper right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()


# -- Graphique average number of customers type for each modality "Nombre de magasin"

g6 = rfm.groupby(['NB_LIGNE_DIFF','SEGMENTv2'])["ID_INDIVIDU"].size().unstack()
# print(g6)
bx = g6.plot(kind='bar',color = ['steelblue', 'lightsteelblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue'])
for p in bx.containers:
    bx.bar_label(p, label_type='edge', fontsize=10)
plt.xlabel("Number of lines",fontweight = 'bold')
plt.ylabel("Number of customers",fontweight = 'bold')
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.title('The average number of customers for each category of the variable "Number of lines".',fontweight = 'bold')
plt.legend(title = "Customers type",loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()



# --------------------------------------------------------------------
# Graphique dimension comportement ----------------------------
# Graphique : Average recency per customers type
g3 = rfm.groupby('SEGMENTv2',as_index=False)['RECENCE'].mean()
# print(g3)
graph1 = plt.bar(g3.SEGMENTv2, g3.RECENCE,color = ['steelblue', 'lightsteelblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue'])
for rect in graph1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
plt.xlabel("Customers type",fontweight = 'bold')
plt.ylabel("Average recency",fontweight = 'bold')
plt.title('Average recency per customers type',fontweight = 'bold')
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()

# print(rfm.columns)


# Graphique : Average frequency per customers type
g4 = rfm.groupby(['FREQUENCE','SEGMENTv2'])["ID_INDIVIDU"].size().unstack()

dx = g4.plot(kind='bar',color = ['steelblue', 'lightsteelblue', 'cornflowerblue', 'dodgerblue', 'deepskyblue'])
for p in dx.containers:
    dx.bar_label(p, label_type='edge', fontsize=10)
plt.xlabel("Number of magasin",fontweight = 'bold')
plt.ylabel("Number of customers",fontweight = 'bold')
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.title('Average frequency per customers type',fontweight = 'bold')
plt.legend(title = "Customers type",loc='upper center')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()


# Montant cumulé moyen 

montant_cumule_median = rfm["MONTANT_CUMULE"].median()
montant_cumule_moyen = rfm["MONTANT_CUMULE"].mean()
# print(montant_cumule_median)
# print(montant_cumule_moyen)




# --------------------------------------------------
# Graphique : Dimension achat

dfs = pd.DataFrame(rfm)
# print(df.describe())
dfs = dfs.dropna(subset=['anciennete'])

data_s = [dfs['anciennete'][dfs['SEGMENTv2'] == 'Weaks'], dfs['anciennete'][dfs['SEGMENTv2'] == 'Lowest-spending customers'],
        dfs['anciennete'][dfs['SEGMENTv2'] == 'Churns'], dfs['anciennete'][dfs['SEGMENTv2'] == 'Good customers'],
        dfs['anciennete'][dfs['SEGMENTv2'] == 'Best customers']]

fig, ay = plt.subplots(figsize=(10, 6))


by = ay.boxplot(data_s, patch_artist=True)

# Définir la couleur des boîtes et des médianes
colors = ['deepskyblue', 'dodgerblue', 'lightsteelblue','cornflowerblue','steelblue']
for patch, color in zip(by['boxes'], colors):
    patch.set_facecolor(color)
for median in by['medians']:
    median.set(color='blue', linewidth=1.5)

# Ajouter des étiquettes aux axes et au titre
ay.set_xlabel('RFM',fontweight = 'bold')
ay.set_ylabel('Days', fontweight = 'bold')
ay.set_title('Boxplot of seniority according to segment', fontweight = 'bold')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()

# Ajouter des étiquettes aux ticks sur l'axe x
ay.set_xticklabels(['Weaks', 'Lowest-spending customers', 'Churns', 'Good customers', 'Best customers'])


# Afficher le graphique
plt.show()

df = pd.DataFrame(rfm)
# print(df.describe())
df = df.dropna(subset=['age'])

data = [df['age'][df['SEGMENTv2'] == 'Weaks'], df['age'][df['SEGMENTv2'] == 'Lowest-spending customers'],
        df['age'][df['SEGMENTv2'] == 'Churns'], df['age'][df['SEGMENTv2'] == 'Good customers'],
        df['age'][df['SEGMENTv2'] == 'Best customers']]

fig, az = plt.subplots(figsize=(10, 6))


bz = az.boxplot(data, patch_artist=True)

# Définir la couleur des boîtes et des médianes
colors = ['deepskyblue', 'dodgerblue', 'lightsteelblue','cornflowerblue','steelblue']
for patch, color in zip(bz['boxes'], colors):
    patch.set_facecolor(color)
for median in bz['medians']:
    median.set(color='blue', linewidth=1.5)

# Ajouter des étiquettes aux axes et au titre
az.set_xlabel('RFM',fontweight = 'bold')
az.set_ylabel('Years', fontweight = 'bold')
az.set_title('Boxplot of age according to segment', fontweight = 'bold')

# Ajouter des étiquettes aux ticks sur l'axe x
az.set_xticklabels(['Weaks', 'Lowest-spending customers', 'Churns', 'Good customers', 'Best customers'])



#Afficher le graphique
plt.show()



# --------------------------------------------------
# Tableau homme/femme

# Compter les valeurs par modalité
counts = df['SEXE'].value_counts()

# Afficher les résultats
# print(counts)



# --------------------------------------------------------------------
# Focus sur les pertes de vitesse   ----------------------------------

# Récence moyenne
masque = g3['SEGMENTv2'] == "Churns"
# print(g3.loc[masque])


# Montant moyen 
masque = g2['SEGMENTv2'] == "Churns"
# print(g2.loc[masque])

# Age moyen
g5 = rfm.groupby('SEGMENTv2',as_index=False)['age'].mean()
masque = g5['SEGMENTv2'] == "Churns"
# print(g5.loc[masque])


# Ancienneté moyenne
g6 = rfm.groupby('SEGMENTv2',as_index=False)['anciennete'].mean()
masque = g6['SEGMENTv2'] == "Churns"
# print(g6.loc[masque])

# Nombre de cadeaux moyen
g7 = rfm.groupby('SEGMENTv2',as_index=False)['NB_CADEAU'].mean()
masque = g7['SEGMENTv2'] == "Churns"
# print(g7.loc[masque])

# Nombre de produits moyen par visite
g8 = rfm.groupby('SEGMENTv2',as_index=False)['NB_PRDT_MOY_VISITE'].mean()
masque = g8['SEGMENTv2'] == "Churns"
# print(g8.loc[masque])


# ----------------------------------------------------------------
# Graphique : Different types of lines purchased by Churns

df_churns = rfm[rfm['SEGMENTv2']=="Churns"]
# print(df_churns)

g7 = df_churns.groupby(['NB_LIGNE_DIFF'],as_index=False)["ID_INDIVIDU"].size()
print(g7)
graph = plt.bar(g7["NB_LIGNE_DIFF"],g7["size"],color="#87CEFA")
plt.xlabel("Number of lines",fontweight = 'bold')
plt.ylabel("Number of customers",fontweight = 'bold')
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.title('Different types of lines purchased by Churns',fontweight = 'bold')
for rect in graph:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.show()



# ----------------------------------------------------------------
# Graphique : Regional dimension

r_Matrice_finale["MER_TERRE"] = r_Matrice_finale["MER_TERRE"].apply(

    lambda x : f'Mer'
        if str(x).lower() == 'mer'
        else x)

g9_a = r_Matrice_finale.groupby(['MER_TERRE'],as_index=False)["CA_MOY_VISITE"].mean()
# print(g9_a)
g9_b = r_Matrice_finale.groupby(['MER_TERRE'],as_index=False)["CA_MOY_VISITE"].median()
# print(g9_b)
g9_c = r_Matrice_finale.groupby(['MER_TERRE'],as_index=False)["CA_MOY_VISITE"].size()
# print(g9_c)

g10_a = r_Matrice_finale.groupby(['REGIONS'],as_index=False)["CA_MOY_VISITE"].mean()
# print(g10_a)
g10_b = r_Matrice_finale.groupby(['REGIONS'],as_index=False)["CA_MOY_VISITE"].median()
# print(g10_b)
g10_c = r_Matrice_finale.groupby(['REGIONS'],as_index=False)["CA_MOY_VISITE"].size()
# print(g10_c)

plt.bar(g9_a["MER_TERRE"], g9_a["CA_MOY_VISITE"],color = ['steelblue','cornflowerblue'])
plt.title('Average amount per region',fontweight = 'bold')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.show()

plt.bar(g10_a["REGIONS"], g10_a["CA_MOY_VISITE"],color = ['steelblue','cornflowerblue'])
plt.title('Average amount per region',fontweight = 'bold')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.tick_params(axis='x', labelsize=10, rotation=0)
plt.tick_params(axis='y', labelsize=10)
plt.show()

