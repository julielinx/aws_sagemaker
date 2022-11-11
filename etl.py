import pandas as pd
import os

if __name__ == '__main__':
    input_path = '/opt/ml/processing/input'
    output_path = '/opt/ml/processing/output'
    
    col_names = ['zip_agg Customer Subtype',
            'zip_agg Number of houses',
            'zip_agg Avg size household',
            'zip_agg Avg age',
            'zip_agg Customer main type',
            'zip_agg Roman catholic',
            'zip_agg Protestant',
            'zip_agg Other religion',
            'zip_agg No religion',
            'zip_agg Married',
            'zip_agg Living together',
            'zip_agg Other relation',
            'zip_agg Singles',
            'zip_agg Household without children',
            'zip_agg Household with children',
            'zip_agg High level education',
            'zip_agg Medium level education',
            'zip_agg Lower level education',
            'zip_agg High status',
            'zip_agg Entrepreneur',
            'zip_agg Farmer',
            'zip_agg Middle management',
            'zip_agg Skilled labourers',
            'zip_agg Unskilled labourers',
            'zip_agg Social class A',
            'zip_agg Social class B1',
            'zip_agg Social class B2',
            'zip_agg Social class C',
            'zip_agg Social class D',
            'zip_agg Rented house',
            'zip_agg Home owners',
            'zip_agg 1 car',
            'zip_agg 2 cars',
            'zip_agg No car',
            'zip_agg National Health Service',
            'zip_agg Private health insurance',
            'zip_agg Income < 30.000',
            'zip_agg Income 30-45.000',
            'zip_agg Income 45-75.000',
            'zip_agg Income 75-122.000',
            'zip_agg Income >123.000',
            'zip_agg Average income',
            'zip_agg Purchasing power class',
            'Contri private third party ins',
            'Contri third party ins (firms)',
            'Contri third party ins (agriculture)',
            'Contri car policies',
            'Contri delivery van policies',
            'Contri motorcycle/scooter policies',
            'Contri lorry policies',
            'Contri trailer policies',
            'Contri tractor policies',
            'Contri agricultural machines policies',
            'Contri moped policies',
            'Contri life ins',
            'Contri private accident ins policies',
            'Contri family accidents ins policies',
            'Contri disability ins policies',
            'Contri fire policies',
            'Contri surfboard policies',
            'Contri boat policies',
            'Contri bicycle policies',
            'Contri property ins policies',
            'Contri ss ins policies',
            'Nbr private third party ins',
            'Nbr third party ins (firms)',
            'Nbr third party ins (agriculture)',
            'Nbr car policies',
            'Nbr delivery van policies',
            'Nbr motorcycle/scooter policies',
            'Nbr lorry policies',
            'Nbr trailer policies',
            'Nbr tractor policies',
            'Nbr agricultural machines policies',
            'Nbr moped policies',
            'Nbr life ins',
            'Nbr private accident ins policies',
            'Nbr family accidents ins policies',
            'Nbr disability ins policies',
            'Nbr fire policies',
            'Nbr surfboard policies',
            'Nbr boat policies',
            'Nbr bicycle policies',
            'Nbr property ins policies',
            'Nbr ss ins policies',
            'Nbr mobile home policies']

    train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test.csv'))
    ground_truth = pd.read_csv(os.path.join(input_path, 'gt.csv'))
    columns = pd.read_csv(os.path.join(input_path, 'col_info.csv'))

    data_dict = {}
    data_dict['feat_info'] = columns.iloc[1:87, 0].str.split(n=2, expand=True)
    data_dict['feat_info'].columns = columns.iloc[0, 0].split(maxsplit=2)
    data_dict['L0'] = columns.iloc[89:130, 0].str.split(n=1, expand=True)
    data_dict['L0'].columns = columns.iloc[88, 0].split()
    data_dict['L2'] = columns.iloc[138:148, 0].str.split(n=1, expand=True)
    data_dict['L2'].columns = ['Value', 'Bin']

    test_df = pd.concat([test, ground_truth], axis=1)
    test_df.columns = data_dict['feat_info']['Name'].to_list()
    train.columns = data_dict['feat_info']['Name'].to_list()

    df = pd.concat([train, test_df], ignore_index=True)
    df.columns = col_names

    data_dict['L0']['Value'] = pd.to_numeric(data_dict['L0']['Value'])
    l0_dict = data_dict['L0'].set_index('Value').to_dict()['Label']
    data_dict['L2']['Value'] = pd.to_numeric(data_dict['L2']['Value'])
    l2_dict = data_dict['L2'].set_index('Value').to_dict()['Bin']
    df[df.columns[0]] = df[df.columns[0]].replace(l0_dict)
    df[df.columns[4]] = df[df.columns[4]].replace(l2_dict)

    df.to_csv(os.path.join(output_path, 'full_data.csv'), index=False)
