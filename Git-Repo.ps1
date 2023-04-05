$url = Read-Host "Introduce la URL del repositorio remoto"

# Inicializar el repositorio local
git init
git add .
git commit -m "Init"
git branch -M master

# Agregar el repositorio remoto y empujar los cambios iniciales
git remote add origin $url
git push -u origin master