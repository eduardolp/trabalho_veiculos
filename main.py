import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

df = pd.read_csv('curva_potencia_fiesta_hatch_2009.txt', sep='\t', decimal=',', header=[0,1])
pot_max  = np.max(df.loc[:, ('Potencia', 'kW')])*1000 # [W]
# print(f'Potência Máxima: {pot_max:.2f} W\n')

# Dados Motor
rendimento_transmissao = 0.88
potencia_cubo = df.loc[:, ('Potencia', 'kW')] * 1000 * rendimento_transmissao # [W]

# Dados Veículo
m = 1076 # [kg] Massa 
C_x = 0.34 # Coeficiente de arrasto
A = 2.02 # [m^2] Área frontal projetada
l = 3.930 # [m] Entre-eixos
h = 0.7255 # [m] Altura CG
x = 0.48 # Distribuição de carga 
mu = 0.850 # Coeficiente de atrito
f = 0.011 # Coeficiente de atrito de rolamento
rho_ar = 1.22557 # [kg/m^3] Massa específica do ar
g = 9.81 # [m/s^2] Gravidade
G = m*g # [N] Peso do veículo

# Dados pneu
r_d = 0.28 # [m]

# Relações de transmissão
relacoes_transmissao = {1:4.083, 
                        2:2.292, 
                        3:1.517, 
                        4:1.108, 
                        5:0.878, 
                        're':3.615}
i_dif = 4.56

curvas_potencia = {i : (potencia_cubo*relacoes_transmissao[i]*i_dif) for i in relacoes_transmissao}
velocidade_marcha_ms = {i : ( 2*np.pi*df.loc[:, ('Rotacao', 'rpm')]*r_d )/ (i_dif*relacoes_transmissao[i]*60) for i in relacoes_transmissao}
# velocidade_marcha_kmh = {i : (df.loc[:, ('Rotacao', 'rpm')]*r_d)/(relacoes_transmissao[i]*i_dif*60) for i in relacoes_transmissao}
vel_max = {i:np.max(velocidade_marcha_ms[i]) for i in velocidade_marcha_ms}
vel_min = {i:np.min(velocidade_marcha_ms[i]) for i in velocidade_marcha_ms}
# print(f'Velocidade máxima: {vel_max}')
# print(f'Velocidade mínima: {vel_min}\n')

######################################################################################################################
#### Resistências ao Movimento
# Aerodinâmica
v_teor = np.linspace(0, 60, 100) # [m/s]
q = 0.5 * rho_ar * v_teor**2 #[kg*m/s^2]
Q_aero = q * C_x * A #[W]

# Inércia - estranho
delta = {i:0.004 + 0.05*relacoes_transmissao[i]**2 for i in relacoes_transmissao}
a = np.linspace(0, 10, 30)
Q_i = {i: m*a*(1+delta[i]) for i in delta} # [N]

# Rolamento
alfa_rol = np.arange(0,90)
Q_r = f*G*np.cos(np.deg2rad(alfa_rol))
Q_r_plano = f*G*np.cos(0)

# Mecânica
Q_m = {i: (1-rendimento_transmissao)*df.loc[:, ('Potencia', 'kW')]*1000/velocidade_marcha_ms[i] for i in velocidade_marcha_ms}

# Aclive
alfa_acl = np.arange(0,90)
Q_s = G * np.sin(np.deg2rad(alfa_acl))
######################################################################################################################

######################################################################################################################
#### Análise do Movimento
# Força motriz máxima
F_mI_max = mu * G * np.cos(0) * ((1-x)+f*(h/l))/(1+mu*(h/l))
# print(f'Força motriz máxima: {F_mI_max:.2f} N')

# Aclive máximo
tan_alfa_max = mu * (((1-x)+f*(h/l))/(1+mu*(h/l))) - f
# print(f'Tangente alfa_max = {tan_alfa_max:.2f}')
alfa_max = np.arctan(tan_alfa_max)
# print(f'Aclive máximo: {alfa_max:.2f} rad = {np.rad2deg(alfa_max):.2f}°\n')

# Aceleração Máxima
ALFA = np.linspace(0, np.rad2deg(alfa_max))
a_max = {i: g/(1+delta[i]) * (((mu*(1-x)-f)-f)/(1+mu*(h/l))*np.cos(np.deg2rad(ALFA))-np.sin(np.deg2rad(ALFA))) for i in delta}
# print(a_max)

######################################################################################################################
#### Diagramas de desempenho
p_consumida = v_teor*(0.5 * rho_ar *v_teor**2+Q_r_plano)

f_consumida = interp1d(v_teor, p_consumida)
potencia_liquida = {i: potencia_cubo - f_consumida(velocidade_marcha_ms[i])  for i in velocidade_marcha_ms}
# print(potencia_liquida[5])

f_quinta = interp1d(velocidade_marcha_ms[5], potencia_liquida[5])

sol = root_scalar(f_quinta, x0=35, x1=45)
# print(f'Velocidade máxima considerando resistências: {sol.root:.2f} m/s') # Velocidade máxima


acel = {i: potencia_liquida[i]/(velocidade_marcha_ms[i]*m*(1+delta[i])) for i in potencia_liquida}
a_max = {i:np.max(acel[i]) for i in acel}
# print(f'Aceleração Máxima: {a_max}')

aclive = {i: np.arcsin(potencia_liquida[i]/velocidade_marcha_ms[i]/G) for i in potencia_liquida}
aclive_deg = {i: np.rad2deg(aclive[i]) for i in aclive}
aclive_max = {i: np.max(aclive_deg[i]) for i in aclive_deg}
# print(f'Aclive máximo: {aclive_max}\n')

######################################################################################################################
#### Cálculo do tempo de retomada
def cria_df(dic, index):
    dic_sem_re = dic.copy()
    dic_sem_re.pop('re')

    df = pd.DataFrame.from_dict(dic_sem_re, orient='columns')
    df['Rotacao'] = index
    df = df.set_index('Rotacao')
    return df

index = df.loc[:, ('Rotacao', 'rpm')]
df_pot = cria_df(potencia_liquida, index)

df_vel = cria_df(velocidade_marcha_ms, index)

df_acel = cria_df(acel, index)

# print(df_pot.head())
# print(df_vel.head())
# print(df_acel.head())


######################################################################################################################
#### Dimensionamento dos Freios

ind_frenagem = ((1-x)+(mu+f)*(h/l))/(x-(mu+f)*(h/l))
# print(f'Índice de Frenagem: {ind_frenagem}')

def disco_dianteiro(ind_frenagem, G, sigma, c, delta_T, delta, vi, vf):
    #ver página 115 da apostila 
    G1_I = sigma * (ind_frenagem*G)/(4*(1+ind_frenagem)*c*delta_T) * (1+delta) * (vi**2-vf**2)
    return G1_I

def disco_traseiro(ind_frenagem, G, sigma, c, delta_T, delta, vi, vf):
    #ver página 115 da apostila 
    G2_I = sigma * G/(4*(1+ind_frenagem)*c*delta_T) * (1+delta) * (vi**2-vf**2)
    return G2_I

def tempo_frenagem(Xi, Omega, vi):
    tempo = 1/np.sqrt(Xi*Omega)*np.arctan(vi*np.sqrt(Xi*Omega))
    return tempo

def dist_frenagem(Xi, Omega, vi):
    s = 1/(2*Xi) * np.log(1+(Xi/Omega)*vi**2)
    return s
    
def Omega(g, delta, mu, f, alpha):
    Omega = g/(1+delta) * ((mu+f)*np.cos(alpha)+np.sin(alpha))
    return Omega

def Xi(m, delta, C_x, A, rho):
    Xi = 1/(2*m*(1+delta))*C_x*A*rho
    return Xi
    
T_amb, T_max = 20, 420 # [°C]
delta_T = T_max-T_amb
vf = 0 # [m/s]
vi = max(vel_max.values()) # [m/s]
delta = 0.05
c = 544.27 # [J/kg/°C]
sigma = 0.99

disco_dianteiro = disco_dianteiro(ind_frenagem, G, sigma, c, delta_T, delta, vi, vf)
disco_traseiro = disco_traseiro(ind_frenagem, G, sigma, c, delta_T, delta, vi, vf)

# print(f'Massa discos dianteiro: {disco_dianteiro/g:.2f} kg')
# print(f'Massa discos traseiro: {disco_traseiro/g:.2f} kg\n')

tempo_imob_80kmh = tempo_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                80/3.6)
dist_frenagem_80kmh = dist_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                80/3.6)

# print(f'Tempo para frenagem (80km/h): {tempo_imob_80kmh:.2f}s')
# print(f'Distância para frenagem (80km/h): {dist_frenagem_80kmh:.2f}m\n')

tempo_imob_100kmh = tempo_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                100/3.6)
dist_frenagem_100kmh = dist_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                100/3.6)

# print(f'Tempo para frenagem (100km/h): {tempo_imob_100kmh:.2f}s')
# print(f'Distância para frenagem (100km/h): {dist_frenagem_100kmh:.2f}m\n')

tempo_imob_vmax = tempo_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                max(vel_max.values()))

dist_frenagem_vmax = dist_frenagem(Xi(m, delta, C_x, A, rho_ar), 
                                Omega(g, delta, mu, f, alpha=0),
                                max(vel_max.values()))

# print(f'Tempo para frenagem ({max(vel_max.values())*3.6:.0f}km/h): {tempo_imob_vmax:.2f}s')
# print(f'Distância para frenagem ({max(vel_max.values())*3.6:.0f}km/h): {dist_frenagem_vmax:.2f}m\n')


######################################################################################################################
#### Gráficos
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax2 = ax1.twinx()

# for i in potencia_liquida_sem_re:
#     ax1.plot(velocidade_marcha_ms[i], potencia_liquida[i], label=f'{i}ª marcha')
# ax1.plot(v_teor, p_consumida, 'k--', label='$P_a+P_r$')
# ax1.plot(df.loc[:, ('Rotacao', 'rpm')],df.loc[:, ('Torque', 'Nm')], color='C0', label='Torque')
# ax1.set_ylim(0,50000)
# ax1.set_xlim(0,50)
# ax1.set_ylabel('Aclive [°]')
# ax1.set_xlabel('Velocidade [m/s]')

# ax2.plot(df.loc[:
# , ('Rotacao', 'rpm')],df.loc[:, ('Potencia', 'kW')], color='C1', label='Potência')
# ax2.set_ylim(0,100)
# ax2.set_ylabel('Potência [kW]', color='C1')
# ax2.set_xlabel('Rotação [rpm]')
# fig.suptitle('Aclive x Velocidade')
# plt.legend()
# plt.show()
# fig.savefig('imagens/aclive.png')