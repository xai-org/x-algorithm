package com.twitter.health.platform_manipulation.user_cred_v2

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec

class UserCredV2AppTest extends AnyWordSpec with Matchers {
  "neutralTeleportMass" should {
    "include every eligible non-near-zero user with equal raw mass" in {
      UserCredV2App.neutralTeleportMass(ValidUserInfo(id = 1L, isNearZero = false)) shouldBe
        Some(UserMass(id = 1L, mass = 1.0))
      UserCredV2App.neutralTeleportMass(ValidUserInfo(id = 2L, isNearZero = false)) shouldBe
        Some(UserMass(id = 2L, mass = 1.0))
    }

    "exclude near-zero users" in {
      UserCredV2App.neutralTeleportMass(ValidUserInfo(id = 3L, isNearZero = true)) shouldBe None
    }
  }

  "sanitizeMass" should {
    "preserve finite positive engagement-derived mass" in {
      UserCredV2App.sanitizeMass(0.25) shouldBe 0.25
    }

    "zero invalid mass before normalization" in {
      Seq(
        0.0,
        -1.0,
        Double.NaN,
        Double.PositiveInfinity,
        Double.NegativeInfinity,
      ).foreach { mass =>
        UserCredV2App.sanitizeMass(mass) shouldBe 0.0
      }
    }
  }
}
