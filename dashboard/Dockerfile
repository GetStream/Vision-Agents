FROM node:24-bookworm-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# NEXT_PUBLIC_* is inlined into the client bundle at build time, so the browser's
# view of the router is fixed here rather than at run time.
ARG NEXT_PUBLIC_ROUTER_URL=http://localhost:8080
ARG NEXT_PUBLIC_CUSTOMER_ID=examples
ENV NEXT_PUBLIC_ROUTER_URL=$NEXT_PUBLIC_ROUTER_URL \
    NEXT_PUBLIC_CUSTOMER_ID=$NEXT_PUBLIC_CUSTOMER_ID

RUN npm run build

FROM node:24-bookworm-slim

WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=3000

COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
