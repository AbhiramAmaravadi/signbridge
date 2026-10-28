# Spring Boot API gateway. Build from this directory:
#   docker build -t signbridge-backend backend/
FROM eclipse-temurin:21-jdk AS build

WORKDIR /app
COPY .mvn .mvn
COPY mvnw pom.xml ./
RUN chmod +x mvnw && ./mvnw -q dependency:go-offline

COPY src src
RUN ./mvnw -q -DskipTests package

FROM eclipse-temurin:21-jre

WORKDIR /app
COPY --from=build /app/target/*.jar app.jar

# Cloud Run sets PORT; application.properties defaults to 8080.
EXPOSE 8080

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
