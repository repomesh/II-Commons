FROM paradedb/paradedb@sha256:788bbbbbc04f7989eba433813347bf11aeeb1b32b57905c6bab07364b3f8411d AS base

# --- Pull vchord.so from vchord image ---
FROM ghcr.io/tensorchord/vchord-postgres:pg17-v0.3.0 AS vchord

# --- Back to main ---
FROM base

# Copy VectorChord plugin (only vchord, not vector)
COPY --from=vchord /usr/lib/postgresql/17/lib/vchord.so /usr/lib/postgresql/17/lib/
COPY --from=vchord /usr/share/postgresql/17/extension/vchord.control /usr/share/postgresql/17/extension/
COPY --from=vchord /usr/share/postgresql/17/extension/vchord--*.sql /usr/share/postgresql/17/extension/

USER root
RUN sed -i "s/\(shared_preload_libraries = 'pg_search,pg_cron\)\('.*\)/\1,vchord\2/" /usr/share/postgresql/postgresql.conf.sample && \
    echo "cron.database_name = 'postgres'" >> /usr/share/postgresql/postgresql.conf.sample
USER postgres
# Expose PostgreSQL port
EXPOSE 5432

# Default entry
CMD ["postgres"]
