import asyncio
import logging
import traceback
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class KerberosConfig(BaseModel):
    keytab_path: str
    principal: str
    renew_interval: int = 3600
    attempts: int = 5
    retry_interval: int = 60


class KerberosRenewer:
    def __init__(self, keytab_path: str, principal: str, **kwargs):
        self._renewer: asyncio.Task | None = None
        self.config = KerberosConfig(
            keytab_path=keytab_path, principal=principal, **kwargs
        )
        if not Path(self.config.keytab_path).exists():
            raise FileNotFoundError(f"Keytab file not found: {self.config.keytab_path}")
        if self.config.renew_interval <= 0:
            raise ValueError("Renew interval must be greater than 0")
        if self.config.attempts <= 0:
            raise ValueError("Attempts must be greater than 0")
        if self.config.retry_interval <= 0:
            raise ValueError("Retry interval must be greater than 0")

    async def _kerberos_renewal_loop(self):
        while True:
            attempts = self.config.attempts
            while attempts > 0:
                attempts -= 1
                try:
                    await self._renew_kerberos_ticket()
                    logger.info(
                        f"Kerberos ticket renewed successfully for {self.config.principal}"
                    )
                    break
                except Exception:
                    if attempts == 0:
                        logger.error(
                            f"Kerberos renewal failed after {self.config.attempts} attempts: {traceback.format_exc()}"
                        )
                    else:
                        logger.warning(
                            f"Kerberos renewal failed, retrying, attempts remaining: {attempts}"
                        )
                    await asyncio.sleep(self.config.retry_interval)
            await asyncio.sleep(self.config.renew_interval)

    async def _renew_kerberos_ticket(self):
        kinit_cmd = [
            "/usr/bin/kinit",
            "-kt",
            self.config.keytab_path,
            self.config.principal,
        ]
        logger.info(
            f"Start renewing Kerberos ticket with command: {' '.join(kinit_cmd)}"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *kinit_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode == 0:
                logger.info(
                    f"Finished renewing Kerberos ticket for {self.config.principal}"
                )
            else:
                logger.error(f"Kerberos renewal failed: {stderr.decode()}")
                raise Exception(f"Kerberos renewal failed: {stderr.decode()}")
        except Exception:
            logger.error(f"Error during Kerberos renewal: {traceback.format_exc()}")
            raise

    async def renew(self):
        await self._renew_kerberos_ticket()

    def start(self):
        logger.info("Starting Kerberos renewer")
        if self._renewer is not None:
            logger.warning("Kerberos renewer already started, skipping")
            return
        self._renewer = asyncio.create_task(self._kerberos_renewal_loop())
        logger.info("Kerberos renewer started")

    def stop(self):
        if self._renewer is not None:
            logger.warning("Stopping Kerberos renewer")
            self._renewer.cancel()
            self._renewer = None
            logger.warning("Kerberos renewer stopped")
        else:
            logger.warning("Kerberos renewer not started, skipping")
