
dfc_date = df.value_counts(['categories', 'yymm']).reset_index().rename(columns={0:'count'}) #
# for ix, row in dfc_date.iterrows():
#     print(row['index'], row['yymm'])
dfc_date = dfc_date.sort_values(by=['categories', 'yymm'], ascending=True)


dfc_date = dfc_date.loc[dfc_date['yymm']!='2022-02']
sns.lineplot(hue='categories', data=dfc_date, x="yymm", y="count", markers=False, style="categories", dashes=False)
plt.xticks(rotation=45)
plt.ylim(dfc_date["count"].min(), 1000)
#plt.xticks(["{}-01-01".format(i) for i in range(2007, 2023)])
plt.title('Overall trend over four categories')
plt.show()





seed_dict = {'superconducting': ['superconduct', 'superconducting loops', 'transmon qubits', 'superconducting qubits'], 
            'ion trap': ['trapped ion', 'ion trap'], 
            'photonics': ['photonic'], 
            'neutral atoms': ['cold atom', 'neutral atom'],
            'silicon': ['spin qubit', 'silicon spin', 'spin quantum computing']}



infos = []
for ix, row in df.iterrows():
    for seed, seed_exp in seed_dict.items():
        if  any([s in row['abstract'] for s in seed_exp]):
            infos.append([row['abstract'], row['yymm'], seed])

dfs = pd.DataFrame(infos, columns=['abstract', 'yymm', 'seed'])
dfs_ = dfs.value_counts(['seed', 'yymm']).reset_index().rename(columns={0:'count'}).sort_values(by=['seed', 'yymm'], ascending=True)


# sns.lineplot( data=dfs_.loc[dfs_['seed']=='superconducting'], x="yymm", y="count", markers=False, dashes=False)

dfs_ = dfs_.loc[dfs_['yymm']!='2022-02']
plt.xticks(rotation=45)
sns.lineplot(hue='seed', data=dfs_, x="yymm", y="count", markers=False, style="seed", dashes=False)
plt.ylim(dfs_["count"].min(), 200)
#plt.xticks(["{}-01".format(i) for i in range(2007, 2023)])
plt.title('Overall trend over four seeds')
plt.show()

plt.savefig("{}.png".format(exp), dpi=1000)
plt.close()




infos = []
for cate in target_categories:
    for yydm in df['yymm'].unique():
        dfa = df.loc[(df['categories']==cate) & (df['yymm']==yydm)]
        if dfa.shape[0] == 0:
            continue 

        authors = []
        for ii in dfa["authors_parsed"].tolist():
            for j in ii:
                authors.append(' '.join(j).strip())
        infos.append([cate, yydm, len(set(authors))])


dfall = pd.DataFrame(infos, columns=['cate', 'yymm','count'])
dfall = dfall.loc[dfall['yymm']!='2022-02']

plt.xticks(rotation=45)
sns.lineplot(hue='cate', data=dfall, x="yymm", y="count", markers=False, style="cate", dashes=False)
plt.ylim(dfall["count"].min(), 1500)
# plt.xticks(["{}-01".format(i) for i in range(2007, 2023)])
plt.title('Overall trend over four authors')
plt.show()

