import gssapi
from gssapi.raw.types import NameType


def patch_gssapi_security_context():
    original_init = gssapi.SecurityContext.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._target_name = gssapi.Name(
            "kafka/kafka@TWITTER.BIZ", name_type=NameType.kerberos_principal
        )

    gssapi.SecurityContext.__init__ = new_init
